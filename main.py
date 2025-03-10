import os
from os.path import dirname, join
from dotenv import load_dotenv
from io import BytesIO
import traceback
import streamlit as st
import openai
import time
from datetime import datetime, timezone
import json
import time
import base64
import re
from functools import reduce
from mimetypes import guess_type
from openai import AssistantEventHandler, AzureOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from typing_extensions import override
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Union
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionChunk
)
from openai.types.beta.threads import (
    TextContentBlock,
    ImageURLContentBlock,
    ImageFileContentBlock,
    ImageFile,
    ImageURL,
    Text,
    Annotation,
    Run
)
import concurrent.futures
import customTools
import serperTools
import internetAccess

@dataclass
class GPTHallucinatedFunctionCall:
    tool_uses: List['HallucinatedToolCalls']
    def __post_init__(self):
        self.tool_uses = [HallucinatedToolCalls(**i) for i in self.tool_uses]

@dataclass
class HallucinatedToolCalls:
    recipient_name: str
    parameters: dict

dotenv_path = join(dirname(__file__), ".env.local")
load_dotenv(dotenv_path)

tools=[{"type": "code_interpreter"}, {"type": "file_search"}, customTools.time, serperTools.run, serperTools.results, serperTools.scholar, serperTools.news, serperTools.places, internetAccess.html]

class StreamHandler(AssistantEventHandler):
    @override
    def __init__(self, client):
        super().__init__()
        self.client = client
        # 親ストリームにコンテンツと最終的なrunを引き渡す（tool_callごとに新しい子ストリームが生じる）
        self.content = []
        self.final_run = None

    @override
    def on_event(self, event):
      if event.event == 'thread.run.requires_action':
          run_id = event.data.id
          self.handle_requires_action(event.data, run_id)

    @override
    def on_image_file_done(self, image_file: ImageFile) -> None:
        print("on_image_file_done ImageFile:", image_file)
        self.content.append(ImageFileContentBlock(type="image_file", image_file=image_file))
        st.image(get_file(image_file.file_id))

    @override
    def on_text_done(self, text: Text) -> None:
        print("on_text_done Text:", text)
        self.content.append(TextContentBlock(type="text", text=text))
        value, files = parse_annotations(text.value, text.annotations)
        put_buttons(files, "stream")

    @override
    def on_tool_call_created(self, tool_call: Any) -> None:
        print(f"\nassistant > tool_call_created > {tool_call.type}\n", flush=True)
        if tool_call.type != "function":
            st.toast(tool_call.type)
        print(tool_call, flush=True)

    @override
    def on_tool_call_delta(self, delta: Any, snapshot: Any) -> None:
        if delta.type == "code_interpreter":
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print("\n\noutput >", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)

    def handle_requires_action(self, data, run_id):
        print(f"\nassistant > {data}\n", flush=True)
        tool_calls = data.required_action.submit_tool_outputs.tool_calls

        tool_outputs = handle_tool_calls(tool_calls)

        with self.client.beta.threads.runs.submit_tool_outputs_stream(
            thread_id=self.current_run.thread_id,
            run_id=self.current_run.id,
            tool_outputs=tool_outputs,
            event_handler=StreamHandler(self.client)
        ) as stream:
            st.write_stream(stream.text_deltas)
            stream.until_done()
        # AssistantEventHandlerに組み込みのイベント機構により、thread.run.completed, canceled, 
        # expired, failed, required_action, incompleteの際に、__current_runが更新され、
        # プロパティcurrent_run()によってアクセスできる
        self.final_run = stream.final_run or stream.current_run
        self.content += stream.content


# メッセージクラスの定義
@dataclass
class ChatMessage:
    role: str
    # contentはAssistant APIのcontent定義を借用
    content: List[Union[ImageFileContentBlock, ImageURLContentBlock, TextContentBlock]]
    files: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

# スレッド管理クラス
class ChatThread:
    def __init__(self, client):
        self.client = client
        self.messages = []
        self.thread_id = None

    def add_message(self, model, role, content, files=None, metadata={}):
        """メッセージを追加"""
        if isinstance(content, str):
            content = [TextContentBlock(type="text", text=Text(value=content, annotations=[]))]

        self.messages.append(ChatMessage(role, content, files, metadata))

        # gpt-4oのAssistant APIで画像認識が出来ない問題はとりあえずペンディングとする
        # Assistant APIはImageURLContentBlockを認識しない？するはずだが・・

        # Assistant APIを未使用の段階ではthread_idは存在しない。初めて使う時に作成して過去のメッセージを登録する。
        # Assistant API時にはレスポンスは自動的にThreadに記録される。
        if self.thread_id and (model["api_mode"] != "assistant" or role != "assistant"):
            self.client.beta.threads.messages.create(
                thread_id = self.thread_id,
                role = role,
                content = self.content_to_content_param(content),
                attachments = files
            )

    def get_last_message(self):
        return self.messages[-1]

    def get_thread_id(self):
        """
        Assistant API用のthread_idを返す。初めてAssistant APIを使うタイミングで
        threadを作成し、過去のメッセージを登録する
        """
        if self.thread_id:
            return self.thread_id

        thread = self.client.beta.threads.create(
            messages = [
                {
                    "role": msg.role,
                    # ToDo: 32メッセージ以上溜まってからだとエラーになる。
                    "content": self.content_to_content_param(msg.content),
                    "attachments": msg.files
                }
                for msg in self.messages
            ]
        )
        self.thread_id = thread.id
        
        return self.thread_id

    @staticmethod
    def content_to_content_param(content: List[Union[ImageFileContentBlock, ImageURLContentBlock, TextContentBlock]]) -> List[dict]:
        """
        オブジェクト形式のcontentを、API送信用にdictに変換する
        """
        content_param = []
        for block in content:
            if block.type == "text":
                # このあたり、微妙に一対一関係ではない
                content_param.append({
                    "type": block.type,
                    "text": block.text.value
                })
            elif block.type == "image_file":
                content_param.append({
                    "type": block.type,
                    "image_file": {"file_id": block.image_file.file_id, "detail": block.image_file.detail},
                })
            elif block.type == "image_url":
                content_param.append({
                    "type": block.type,
                    "image_url": {"url": block.image_url.url, "detail": block.image_url.detail},
                })
            else:
                raise ValueError(f"未知のコンテンツブロックの type: {block.type}")
        return content_param

# セッション管理クラス
class ConversationManager:
    def __init__(self, clients, assistants):
        self.client = clients["openai"]
        self.thread = ChatThread(self.client)
        self.assistants = assistants

    def add_message(self, model, role, content, files=None, metadata={}):
        """メッセージをChatThreadに追加"""
        self.thread.add_message(model, role, content, files, metadata)

    def get_completion_messages(self, model, text_only=False):
        """Completion API用にメッセージを変換"""
        messages = []
        for msg in self.thread.messages:
            # Assistant API用のImageFileContentBlockは除く
            content = [cont for cont in msg.content if not isinstance(cont, ImageFileContentBlock)]

            # Visionサポートの無いモデルにImageを与えるとエラーになるので除く
            if not model.get("support_vision", False):
                content = [cont for cont in content if isinstance(cont, TextContentBlock)]

            if text_only:
                # Deepseekなど、テキストだけ必要な場合はテキストを抽出する
                content = "\n".join([cont.text.value for cont in content if isinstance(cont, TextContentBlock)])
            else:
                # そうでない場合はclassからdictに変換する
                content = self.thread.content_to_content_param(content)

            messages.append({
                "role": "assistant" if msg.role == "assistant" else "user",
                "content": content
            })
        return messages

    def create_attachments_for_assistant(self, files, tool_for_files):
        """Assistant用のファイルアップロード"""
        attachments = []
        for file in files:
            file.seek(0)
            response = self.client.files.create(
                file=file,
                purpose="assistants"
            )
            attachments.append(
                    {
                        "file_id": response.id,
                        "tools": [{"type": tool_for_files}],
                    }
            )
        return attachments

    def create_ImageURLContentBlock(self, file, detail_level):
        mime_type = guess_type(file.name)[0]
        image_encoded = base64.b64encode(file.getvalue()).decode()
        image_url = f'data:{mime_type};base64,{image_encoded}'
        return ImageURLContentBlock(
            type="image_url",
            image_url=ImageURL(url=image_url, detail=detail_level)
        )

# gpt-4oがAssistant APIで画像を認識しないので、ImageFileならと思って加えたが、どうやらモデルの方の問題らしい
#    def create_ImageFileContentBlock(self, file, detail_level):
#        response = self.client.files.create(
#            file=file,
#            purpose="vision"
#            # "vision"を指定すると、"purpose contains an invalid purpose vision"と言われてしまう。
#            # AzureのAPIが追いついていない可能性あり。"assistant"なら受け付けるが、gpt-4oは自分には画像認識能力が無いと言う
#        )
#        return ImageFileContentBlock(
#            type="image_file",
#            image_file=ImageFile(file_id=response.id, detail=detail_level)
#        )

def pretty_print(messages: List[ChatMessage]) -> None:
    i = -1
    m = None
    for i0, m0 in enumerate(messages):
#        print("role:", m.role)
#        print("content:", m.content)
        if i != -1:
            with st.chat_message("assistant" if m.role == "assistant" else "user"):
                pretty_print_message(i, m)
        i = i0
        m = m0

    # 最新のメッセージのみtoken_summaryを表示
    if i != -1:
        with st.chat_message("assistant" if m.role == "assistant" else "user"):
            pretty_print_message(i, m, with_token_summary=True)


def pretty_print_message(key, message, with_token_summary=False):
    for j, cont in enumerate(message.content):
        if isinstance(cont, ImageFileContentBlock) and message.role == "assistant":
            # 自分でアップロードしたファイルをダウンロードしようとするとエラーとなるので、asssistantの場合のみ表示
            st.image(get_file(cont.image_file.file_id))
        if isinstance(cont, ImageURLContentBlock):
            st.image(cont.image_url.url)
        if isinstance(cont, TextContentBlock):
            value, files = parse_annotations(cont.text.value, cont.text.annotations)
            st.markdown(value, unsafe_allow_html=True)
            put_buttons(files, f"hist{key}-{j}")
    print(message)
    print(with_token_summary)
    if "file_search_results" in message.metadata:
        put_quotations(message.content, message.metadata["file_search_results"])
    if with_token_summary and "token_summary" in message.metadata:
        st.markdown(message.metadata["token_summary"])

def put_buttons(files, key=None) -> None:
    for i, file in enumerate(files):
        if key:
            key=f"{key}-{i}"
        else:
            key = None
        if file["type"] == "file_path":
            st.download_button(
                f"{file["index"]}: {file["filename"]} : ダウンロード",
                get_file(file["file_id"]),
                file_name=file["filename"],
                key=key
            )

def put_quotations(content, file_search_results):
    citations = {}
    for cont in content:
        if isinstance(cont, TextContentBlock):
            for annotation in cont.text.annotations:
                if annotation.type == "file_citation" and (match := re.search(r'(\d+):(\d+)', annotation.text)):
                    i = int(match[2])
                    if i not in citations:
                        citations[i] = annotation.text
    for i, text in citations.items():
        result = file_search_results[i]
        with st.expander(f"{text}: {result.file_name}"):
            st.write(f"""
~~~
score: {result.score}
~~~
{result.content[0].text}
""")

def get_file(file_id: str) -> bytes:
    key = f"content_{file_id}"
    if key in st.session_state.fileCache:
        return st.session_state.fileCache[key]

    client = st.session_state.clients["openai"]
    retrieve_file = client.files.with_raw_response.content(file_id)
    content: bytes = retrieve_file.content
    st.session_state.fileCache[key] = content
    return content

def get_file_info(file_id: str) -> bytes:
    key = f"info_{file_id}"
    if key in st.session_state.fileCache:
        return st.session_state.fileCache[key]

    client = st.session_state.clients["openai"]
    retrieve_file = client.files.retrieve(file_id)
    st.session_state.fileCache[key] = retrieve_file
    return retrieve_file

def parse_annotations(value: str, annotations: List[Annotation]):
    files = []
    print(value)
    print(annotations)
    for (
        index,
        annotation,
    ) in enumerate(annotations):
        # FilePathAnnotation
        if annotation.type == "file_path":
            files.append(
                {
                    "type": annotation.type,
                    "file_id": annotation.file_path.file_id,
                    "filename": annotation.text.split("/")[-1],
                    "text": annotation.text,
                    "index": index
                }
            )
        elif annotation.type == "file_citation":
            info = get_file_info(annotation.file_citation.file_id)
            files.append(
                {
                    "type": annotation.type,
                    "file_id": annotation.file_citation.file_id,
                    "filename": info.filename,
                    "text": annotation.text,
                    "index": index
                }
            )
    value = re.sub(r'\[([^\]]*)\]\((sandbox:[^)]*)\)', r'ボタン\1', value)
    return value, files

def handle_tool_calls(tool_calls: List[Dict], mode = "assistant") -> List[Dict]:
    """
    ツール呼び出しを処理する関数
    """
    print(tool_calls)
    def add_output(outputs, tool_call_id, output, mode, fname = None):
        if mode == "assistant":
            # Assistant API用のtool_output
            outputs.append({
                "tool_call_id": tool_call_id,
                "output": output
            })
        else:
            # Completion API用のtool_output
            outputs.append({
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": fname,
                "content": output,
            })

    tool_outputs = []

    for tool in tool_calls:
        function = tool.function
        fname = function.name
# 並列ツール呼び出しのテスト用データ（テスト時に都合よく発生してくれないので）
#        function.name = "multi_tool_use.parallel"
#        function.arguments = '{"tool_uses": [{"recipient_name": "functions.get_google_results", "parameters": {"query": "OpenAI O1 processor"}}, {"recipient_name": "functions.get_google_results", "parameters": {"query": "OpenAI O1 chip"}}]}'
        fargs = json.loads(function.arguments)
        print(f"Function call: {fname}")
        print(f"Function arguments: {fargs}")

        if function.name == "multi_tool_use.parallel":
            # 並列ツール呼び出し：AIが並列実行を要求してくることがある。
            # 仕様外の動作(ハルシネーション)だという説もある。動作検証未済。
            # We need to deserialize the arguments
            caught_calls = GPTHallucinatedFunctionCall(**(json.loads(function.arguments)))
            tool_uses = caught_calls.tool_uses

            # ThreadPoolExecutorで並列実行
            with concurrent.futures.ThreadPoolExecutor() as executor:

                # 各ツール呼び出しを実行用タスクに変換
                future_to_tool = {
                    executor.submit(
                        function_calling,
                        tool_use.recipient_name.rsplit('.', 1)[-1],
                        tool_use.parameters
                    ): {"id": tool.id, "fname": tool_use.recipient_name.rsplit('.', 1)[-1]}
                    for tool_use in tool_uses
                }

                # 完了したタスクから結果を収集
                results = []
                for future in concurrent.futures.as_completed(future_to_tool):
                    tool_call_id = future_to_tool[future]["id"]
                    fname = future_to_tool[future]["fname"]
                    print(f"fname: {fname}")
                    print(f"tool_call_id: {tool_call_id}")
                    try:
                        results.append(future.result())
                    except Exception as e:
                        print(f"Error: {str(e)}")
                        results.append(json.dumps(f"Error: {str(e)}"))

                print(results)
                # 並列ツール呼び出しについて、tool_call_idは一つしかなく、jsonを改行で連結し一つの
                # tool_outputにまとめて返す(jsonl?)。この情報はコミュニティの会話より得たが正しいか分からない。
                add_output(tool_outputs, tool_call_id, "\n".join(results), mode, fname)
        else:
              # 順次ツール呼び出し
              fresponse = function_calling(fname, fargs)
              add_output(tool_outputs, tool.id, fresponse, mode, fname)

    print("tool_outputs:")
    print(tool_outputs)
    return tool_outputs

def function_calling(fname, fargs):
        print(f"fc fname: {fname}")
        print(f"fc fargs: {fargs}")
        if fname == "get_current_weather":
            fresponse = customTools.get_current_weather(
                location=fargs.get("location"),
            )
        elif fname == "get_current_datetime":
            st.toast("[datetime]", icon="🕒");
            fresponse = customTools.get_current_datetime(
                timezone=fargs.get("timezone")
            )
        elif fname == "get_google_serper":
            st.toast(f"[Google Serper] {fargs.get("query")}", icon="🔍");
            fresponse = serperTools.get_google_serper(
                query=fargs.get("query")
            )
        elif fname == "get_google_results":
            st.toast(f"[Google detail] {fargs.get("query")}", icon="🔍");
            fresponse = serperTools.get_google_results(
                query=fargs.get("query")
            )
        elif fname == "get_google_scholar":
            st.toast(f"[Google scholar] {fargs.get("query")}", icon="🎓");
            fresponse = serperTools.get_google_scholar(
                query=fargs.get("query")
            )
        elif fname == "get_google_news":
            st.toast(f"[Google news] {fargs.get("query")}", icon="📰");
            fresponse = serperTools.get_google_news(
                query=fargs.get("query")
            )
        elif fname == "get_google_places":
            st.toast(f"[Google places] {fargs.get("query")}", icon="🍽️");
            fresponse = serperTools.get_google_places(
                query=fargs.get("query"),
                country=fargs.get("country", "jp"),
                language=fargs.get("language", "ja")
            )
        elif fname == "parse_html_content":
            st.toast("[parse html content]", icon="👀");
            fresponse = internetAccess.parse_html_content(
                url=fargs.get("url"),
                query=fargs.get("query", "headings"),
                heading=fargs.get("heading", None)
            )
        return fresponse

# API実行モジュール
def execute_api(model, selected_tools, conversation, options = {}):

    thread = conversation.thread
    client = model["client"]

    if model["api_mode"] == "inference":
        # DeepSeekやPhi向けのInference API
        # https://learn.microsoft.com/en-us/rest/api/aifoundry/modelinference/
        messages = conversation.get_completion_messages(model, text_only=True)
        print(messages)
        try:
            if model["streaming"]:
                response = client.complete({
                    "stream": True,
                    "messages": messages,
# 空のtoolsを与えただけでも不安定になる?いや、toolsを与えなくてもこのエラーは出ることがある。サーバー側の混雑状況によるのではないか？ DeepSeek APIエラー: Operation returned an invalid status 'Too Many Requests' Content: Please check this guide to understand why this error code might have been returned
#                    "tools": [],
# 現時点でfunction callingはサポートされていない。toolsを指定すると、反応が止まってしまう。2025/2
# https://github.com/deepseek-ai/DeepSeek-R1/issues/9
#                    "tools": [customTools.time, serperTools.run, serperTools.results, serperTools.news, serperTools.places, internetAccess.html],
                    "model": model["model"],
                    "max_tokens": 4096
                })
                print(response)
                digester = completion_streaming_digester(response)
                full_response = st.write_stream(digester.generator)
                response = digester.response
                response_message = ChatCompletionMessage.model_validate(response["choices"][0])
                print(response)
                full_response = response_message.content
            else:
                response = client.complete({
                    "messages": messages,
                    "max_tokens": 4096
                })
                full_response = response.choices[0].message.content

            token_summary = get_token_summary(response, model)
            st.markdown(token_summary)
            metadata = {"token_summary": token_summary}
            return full_response, metadata

        except Exception as e:
            st.error(f"Azure AI Model Inference APIエラー")
            raise

    elif model["api_mode"] == "assistant":
        thread_id = conversation.thread.get_thread_id()
        print(thread_id)

        args = {"thread_id": thread_id, "assistant_id": model["assistant_id"]} | options

        if model["support_tools"]: # selected_toolsが空の場合もassistant設定を上書き
            args["tools"] = selected_tools
            print(f"selected tools: {selected_tools}")

        if model["streaming"]:
            # ストリーミング対応のAssistant API実行
            args["event_handler"] = StreamHandler(client)
            try:
                print(args)
                with client.beta.threads.runs.stream(**args) as stream:
                    st.write_stream(stream.text_deltas)
                    stream.until_done()

                run = stream.final_run or stream.current_run
                content = stream.content
                print(content)
                print(run)
                file_search_results = get_file_search_results(thread_id, run.id)
                put_quotations(content, file_search_results)
                token_summary = get_token_summary(run, model)
                st.markdown(token_summary)
                metadata = {"token_summary": token_summary, "file_search_results": file_search_results}
                return content, metadata

            except Exception as e:
                st.error(f"Assistant(streaming) APIエラー")
                raise

        else:
            # 非ストリーミング版 Assistant APIの実行
            try:
                # 実行開始
                run = client.beta.threads.runs.create(**args)

                # 実行完了を待機
                while run.status not in ["completed", "failed"]:
                    print(run)
                    time.sleep(1)
                    run = client.beta.threads.runs.retrieve(
                        thread_id=thread.thread_id,
                        run_id=run.id
                    )
                    if run.status == "requires_action":
                        print(run.required_action)
                        messages = handle_tool_calls(run.required_action.submit_tool_outputs.tool_calls, "assistant")
                        run = client.beta.threads.runs.submit_tool_outputs(
                          thread_id=thread.thread_id,
                          run_id=run.id,
                          tool_outputs=messages
                        )

                if run.status == "failed":
                    raise Exception(f"実行に失敗しました: {run.last_error}")

                # レスポンスを取得
                print(run)
                messages = client.beta.threads.messages.list(
                    thread_id=thread.thread_id,
                    limit=1
                )
                print(messages)
                content = messages.data[0].content
                pretty_print_message("assist_msg", messages.data[0])
                token_summary = get_token_summary(run, model)
                st.markdown(token_summary)
                metadata = {"token_summary": token_summary}
                return content, metadata

            except Exception as e:
                st.error(f"Assistant APIエラー")
                raise

    else:
        # Completion API実行処理
        messages = conversation.get_completion_messages(model)
        try:
            args = {"model": model["model"], "messages": messages} | options

            print(model)
            if model["streaming"]:
                args["stream"] = True
                args["stream_options"] = {"include_usage": True}

            if model["support_tools"] and selected_tools:
                args["tools"] = selected_tools
                print(f"selected tools: {selected_tools}")

            full_response = ""
            while True:
                print(f"args: {args}")
                response = client.chat.completions.create(**args)
                print(response)

                if model["streaming"]:
                    # ストリーミング対応のCompletion API実行
                    digester = completion_streaming_digester(response)
                    full_response += st.write_stream(digester.generator)
                    response = digester.response
                    print(response)
                    response_message = ChatCompletionMessage.model_validate(response["choices"][0])

                else:
                    # 非ストリーミング対応のCompletion API実行
                    response_message = response.choices[0].message
                    if hasattr(response_message, "content") and response_message.content:
                        st.write(response_message.content)
                        full_response += response_message.content
                    print(response)

                messages.append(response_message.model_dump())

                if hasattr(response_message, "tool_calls") and response_message.tool_calls:
                    messages += handle_tool_calls(response_message.tool_calls, "completion")

                else:
                    break
                
            token_summary = get_token_summary(response, model)
            st.markdown(token_summary)
            metadata = {"token_summary": token_summary}
            return full_response, metadata

        except Exception as e:
            st.error(f"Completion APIエラー")
            raise

def get_file_search_results(thread_id, run_id):
    client = st.session_state.clients["openai"]

    # Run Stepを取得
    run_steps = client.beta.threads.runs.steps.list(
        thread_id=thread_id,
        run_id=run_id
    )

    # 最後のRun StepからFile Searchの実行結果を含めて取得する
    run_step = client.beta.threads.runs.steps.retrieve(
        thread_id=thread_id,
        run_id=run_id,
        step_id=run_steps.data[-1].id,
        include=["step_details.tool_calls[*].file_search.results[*].content"]
    )
    if hasattr(run_step.step_details, "tool_calls"):
        file_search_tcs = [tc for tc in run_step.step_details.tool_calls if hasattr(tc, "file_search")]
        if file_search_tcs:
            return file_search_tcs[0].file_search.results
    print(run_step)
    return []

def get_token_summary(response, model):
    if isinstance(response, ChatCompletion) or isinstance(response, Run):
        response = response.model_dump()
    token_summary = ""
    if "usage" in response and reduce(
        lambda a, c:c in response["usage"] and a,
        ["completion_tokens", "prompt_tokens", "total_tokens"], True):
        usage = response["usage"]
        token_summary = f"tokens in:{usage["prompt_tokens"]} out:{usage["completion_tokens"]} total:{usage["total_tokens"]}"
        if "pricing" in model:
            token_summary += f" cost: US${(usage["prompt_tokens"] * model["pricing"]["in"] + usage["completion_tokens"] * model["pricing"]["out"]) / 1000000}"
        token_summary = f"\n:violet-background[{token_summary}]"

    return token_summary 

class completion_streaming_digester:
    def __init__(self, stream):
        self.stream = stream
        self.response = {}

    def generator(self):
    # メッセージ及びtool_callの断片を受け取りながらUIに反映し、最終的に完全なメッセージを復元する
        for chunk in self.stream:
            print(chunk)

            if isinstance(chunk, ChatCompletionChunk):
                # OpenAIのCompltion APIの場合
                chunk_dict = chunk.model_dump()
            else:
                # deepseekなど
                chunk_dict = chunk

            for key, value in chunk_dict.items():
                if key == "choices":
                    if key not in self.response:
                        self.response[key] = {}
                    for choice in value:
                        for ckey, cvalue in choice.items():
                            ci = choice["index"]
                            if ci not in self.response[key]:
                                self.response[key][ci] = {"content": "", "tool_calls": {}}
                            if ckey == "delta":
                                for dkey, dvalue in cvalue.items():
                                    if dkey == "content" and dvalue:
                                        self.response[key][ci][dkey] += dvalue
                                        # UIに出力
                                        yield dvalue
                                    elif dkey == "tool_calls" and dvalue:
                                        for tool_call in dvalue:
                                            for tkey, tvalue in tool_call.items():
                                                ti = tool_call["index"]
                                                if ti not in self.response[key][ci][dkey]:
                                                    self.response[key][ci][dkey][ti] = {"function": {"arguments":""}}
                                                if tkey == "function" and tvalue:
                                                    if "name" in tvalue and tvalue["name"]:
                                                        self.response[key][ci][dkey][ti][tkey]["name"] = tvalue["name"]
                                                    if "arguments" in tvalue and tvalue["arguments"]:
                                                        self.response[key][ci][dkey][ti][tkey]["arguments"] += tvalue["arguments"]
                                                elif tvalue:
                                                    self.response[key][ci][dkey][ti][tkey] = tvalue
                                    elif dvalue:
                                        self.response[key][ci][dkey] = dvalue
                            elif cvalue:
                                self.response[key][ci][ckey] = cvalue
                elif value:
                    self.response[key] = value
        # 上記マージ作業の都合上dictで表現された配列をlistに変換する
        choices = []
        print(self.response)
        for ci, cvalue in sorted(self.response["choices"].items(), key=lambda x:x[0]):
            tool_calls = []
            for ti, tvalue in sorted(cvalue["tool_calls"].items(), key=lambda x:x[0]):
                tool_calls.append(tvalue)
            cvalue["tool_calls"] = tool_calls
            choices.append(cvalue)
        self.response["choices"] = choices

def get_assistant(client, mode):
    # IF: https://platform.openai.com/docs/assistants/how-it-works/creating-assistants
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00:00")

    if mode == "development":
        instructions=f"あなたは汎用的なアシスタントです。質問には簡潔かつ正確に答えてください。現在の日時は「{current_time}」であることを考慮し、時機にかなった回答を心がけます。あなたはオンラインで最新の情報を検索することができます。"
        tools=[{"type": "code_interpreter"}, customTools.time, serperTools.run, serperTools.results, serperTools.news, serperTools.places, internetAccess.html]
    else:
        instructions=f"あなたは汎用的なアシスタントです。質問には簡潔かつ正確に答えてください。現在の日時は「{current_time}」であることを考慮し、時機にかなった回答を心がけます。あなたはオンラインで最新の情報を検索することができます。"
        tools=[{"type": "code_interpreter"}, customTools.time, serperTools.run, serperTools.results, internetAccess.html]

    name=f"汎用アシスタント({mode})"
    assistant = None
    assistants = client.beta.assistants.list(order='desc', limit="100")
    print(assistants)
    for i in assistants.data:
        if i.created_at < time.time() - 86400:
            client.beta.assistants.delete(assistant_id=i.id)
            time.sleep(.2)
        elif i.name == name:
            assistant = i

    if not assistant:
        assistant = client.beta.assistants.create(
            name=name,
            model="gpt-4o"
        )

    client.beta.assistants.update(
        assistant_id=assistant.id,
        instructions=instructions,
        tools=tools
    )

    return assistant.id

# 初期化
if "clients" not in st.session_state:
    st.session_state.clients = {
        "openai": AzureOpenAI(
            azure_endpoint = os.getenv("ENDPOINT_URL"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-12-01-preview"
        ),
        "deepseek": ChatCompletionsClient(
            endpoint=os.getenv("DEEPSEEK_ENDPOINT_URL"),
            credential=AzureKeyCredential(os.getenv("DEEPSEEK_AZURE_INFERENCE_CREDENTIAL"))
        ),
        "phi4": ChatCompletionsClient(
            endpoint=os.getenv("PHI_4_ENDPOINT_URL"),
            credential=AzureKeyCredential(os.getenv("PHI_4_AZURE_INFERENCE_CREDENTIAL"))
        )
    }

if "assistants" not in st.session_state:
    st.session_state.assistants = {
        "gpt-4o": get_assistant(st.session_state.clients["openai"], os.getenv("IASA_DEPLOYMENT_MODE", "development"))
    }

if "conversation" not in st.session_state:
    st.session_state.conversation = ConversationManager(st.session_state.clients, st.session_state.assistants)
conversation = st.session_state.conversation 

models = {
  "o3-mini": {
    "model": "o3-mini",
    "client": st.session_state.clients["openai"],
    "api_mode": "completion",
    "support_vision": False,
    "support_tools": True,
    "support_reasoning_effort": True,
    "streaming": True,
    "pricing": {"in": 1.1, "out":4.4}
  },
  "GPT-4o": {
    "model": "gpt-4o",
    "client": st.session_state.clients["openai"],
    "api_mode": "assistant",
    "assistant_id": st.session_state.assistants["gpt-4o"],
    "support_vision": False,
    "support_tools": True,
    "streaming": True,
    "pricing": {"in": 2.5, "out":10}
  },
#  "GPT-4o-nostream": {
#    "model": "gpt-4o",
#    "client": st.session_state.clients["openai"],
#    "api_mode": "assistant",
#    "assistant_id": st.session_state.assistants["gpt-4o"],
#    "support_vision": False,
#    "support_tools": True,
#    "streaming": False,
#    "pricing": {"in": 2.5, "out":10}
#  },
  "GPT-4o-completion": {
    "model": "gpt-4o",
    "client": st.session_state.clients["openai"],
    "api_mode": "completion",
    "assistant_id": st.session_state.assistants["gpt-4o"],
    "support_vision": True,
    "support_tools": True,
    "streaming": True,
    "pricing": {"in": 2.5, "out":10}
  },
  "o1": {
    "model": "o1",
    "client": st.session_state.clients["openai"],
    "api_mode": "completion",
    "support_vision": True,
    "support_tools": True,
    "support_reasoning_effort": True,
    "streaming": False,
    "pricing": {"in": 15, "out":60}
  },
  "DeepSeek R1": {
    "model": "DeepSeek-R1",
    "client": st.session_state.clients["deepseek"],
    "api_mode": "inference",
    "streaming": True,
    "pricing": {"in": 0, "out":0}
  },
  "Microsoft Phi-4": {
    "model": "Phi-4",
    "client": st.session_state.clients["phi4"],
    "api_mode": "inference",
    "streaming": True,
    # これはEastUS2の価格であり、正確ではない
    # https://techcommunity.microsoft.com/blog/machinelearningblog/affordable-innovation-unveiling-the-pricing-of-phi-3-slms-on-models-as-a-service/4156495
    "pricing": {"in": 0.125, "out": 0.5}
  }
}

if "fileCache" not in st.session_state:
    st.session_state.fileCache = {}
if "processing" not in st.session_state:
    st.session_state.processing = False
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0
if 'switches' not in st.session_state:
    st.session_state.switches = {
        "code_interpreter": True,
        "file_search": True,
        "get_google_results": True,
        "parse_html_content": True
    }

# メインUI
st.title("Dual API Chat Interface")

# サイドバー設定
with st.sidebar:
    model = models[st.selectbox(
        "Model",
        models.keys()
    )]
    options = {}

    st.text("API Mode: " + model["api_mode"])
    st.text("Streaming: " + ("True" if model.get("streaming", False) else "False"))
    st.text("Support vision: " + ("True" if model.get("support_vision", False) else "False"))

    if model.get("support_reasoning_effort", False):
        options["reasoning_effort"] = st.selectbox(
            "reasoning_effort",
            ["low", "medium", "high"],
            index = 2
        )

    uploaded_files = None
    image_files = None
    if model["api_mode"] == "assistant":
        uploaded_files = st.file_uploader(
            "ファイルアップロード",
            accept_multiple_files=True,
            key = f"file_uploader_{st.session_state.uploader_key}"
        )
        tool_for_files = st.selectbox(
            "ファイルの用途",
            ["file_search", "code_interpreter"]
        )
        st.session_state.switches[tool_for_files] = True

    if model.get("support_vision", False):
        image_files = st.file_uploader(
            "画像ファイル",
            accept_multiple_files=True,
            type = ["png", "jpeg", "jpg", "webp", "gif"],
            key = f"image_uploader_{st.session_state.uploader_key}"
        )
        detail_level = st.selectbox(
            "画像の詳細レベル",
            ["auto", "high", "low"]
        )

    # "code_interpreter", "file_search"を使えるのはassistantの時だけ
    if model["api_mode"] != "assistant":
        tools = [tool for tool in tools if tool.get("type", None) == "function"]

    # 表示用ラベル生成
    tool_names = [
        t.get("type") if t.get("type") != "function"
        else t["function"]["name"]
        for t in tools
    ]

    # サイドバーにtool選択用トグルスイッチを表示
    if model.get("support_tools", False):
        st.header("ツール選択")
        switches = {
            name: st.toggle(
                name,
                value=st.session_state.switches[name] if name in st.session_state.switches else False,
                key=f"tool_switch_{name}"
            )
            for name in tool_names
        }
        st.session_state.switches = st.session_state.switches | switches

        # 選択に合わせツールをフィルタリング
        selected_tools = [
            tool
            for i, tool in enumerate(tools)
            if st.session_state.switches[tool_names[i]]
        ]
    else:
        selected_tools = []
    print(selected_tools)

if content := st.chat_input("メッセージを入力"):
        # ファイル処理
        attachments = []
        if uploaded_files:
            st.toast("ファイルをアップロードします")
            attachments = conversation.create_attachments_for_assistant(uploaded_files, tool_for_files)
            st.toast("アップロード完了")
            st.session_state.uploader_key += 1 # 選択されたファイルをクリア

        if image_files:
            st.toast("画像をアップロードします")
            content = [TextContentBlock(type="text", text=Text(value=content, annotations=[]))]
            for file in image_files:
                # ImageURLが、ImageFileの両方作成しておき、API実行時に不要な方は捨てる
                # ただし、Thread未作成時にはImageURLのみ作成とし、Thread作成時にImageURLから変換生成する
                # という計画だったが、どうやらgpt-4oはAssistant API時には画像を認識できない模様（本人談）
                # 現状、Assistant API, Vision両対応のモデルが他にないので、一旦あきらめ、ImageURLのみ対応。
                content.append(conversation.create_ImageURLContentBlock(file,detail_level))
            st.toast("アップロード完了")
            st.session_state.uploader_key += 1 # 選択されたファイルをクリア
    
        # メッセージ追加
        conversation.add_message(model, "user", content, attachments)
        # 処理中へ移行
        st.session_state.processing = True
        st.rerun()  # ここで一旦再描画(ファイルのクリアなどに必要)
    
# メッセージ表示
pretty_print(conversation.thread.messages)

if st.session_state.get("processing"):
    # 処理すべきユーザーメッセージがある

    # API実行
    try:
        # アシスタントメッセージのプレースホルダーを作成
        with st.chat_message("assistant"):

            content, metadata = execute_api(
                model,
                selected_tools,
                conversation,
                options
                )
        
            # レスポンスを履歴に追加
            conversation.add_message(model, "assistant", content, None, metadata)
            st.session_state.processing = False

            # 最終的な内容で描画しなおすべきか？当面不要と判断。
            # placeholderを使って清書する事も考えたが、ラウザ側との同期に失敗し、前のコンテナのコンテンツが
            # 残ってしまい、これはうまくいかない。
            # streamlitでは既に描画したものを変更するのはやめた方がいいかも。
            # どうしても再描画が必要なら(引用番号の書き換えなど)、素直にrerun()した方がいい。
#        st.rerun()
    
    except Exception as e:
        st.error(f"処理中にエラーが発生しました")
        st.exception(e)
        st.button("リトライ")
