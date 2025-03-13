import json
import requests
from bs4 import BeautifulSoup

html = {
    "type": "function",
    "function": {
        "name": "parse_html_content",
        "description": "指定されたURLからHTMLコンテンツを取得し、特定の情報を抽出する。特定のヘッディングセクションのテキストも抽出可能。",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "情報を抽出するウェブページのURL",
                },
                "query": {
                    "type": "string",
                    "description": "抽出する情報の種類を指定。metadata_and_headings, section_html, html, text, links が利用可能。",
                    "enum": ["metadata_and_headings", "section_html", "html", "text", "links"]
                },
                "heading": {
                    "type": "string",
                    "description": "特定のヘッディングセクションからのテキストを抽出するために必要。queryがsection_textまたはsection_htmlの場合に使用。",
                    "nullable": True
                }
            },
            "required": ["url", "query"]
        }
    }
}

def fetch_page_content(url):
    """
    指定されたURLからページ内容を取得する
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # エラーがあれば発生させる
        return response.text
    except requests.exceptions.ConnectionError:
        return "接続エラーが発生しました。URLが正しいことを確認してください。"
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return "URLにアクセスしたところ404 Not Foundでした。"
        else:
            return f"HTTPエラーが発生しました: {response.status_code}"
    except requests.exceptions.Timeout as e:
        return f"リクエストがタイムアウトしました: {e}"
    except requests.exceptions.TooManyRedirects as e:
        return "リダイレクト回数が多すぎる"
    except requests.exceptions.InvalidURL as e:
        return "URL の形式が不正"
    except requests.exceptions.MissingSchema as e:
        return "URL にスキーマ (http, https など) が含まれていない"
    except requests.exceptions.InvalidHeader as e:
        return "不正なヘッダーが指定された"
    except requests.exceptions.ContentDecodingError as e:
        return "レスポンスボディのデコードに失敗した"
    except requests.exceptions.RequestException as e:
        return f"その他のエラーが発生しました: {e}"

def parse_html_content(url, query="", heading=None):
    """
    指定されたURLからHTMLコンテンツを取得し、特定の情報を抽出する
    特定のヘッディングセクションのテキストも抽出可能
    """
    html_content = fetch_page_content(url)
    if "404 Not Found" in html_content:
        return html_content

    soup = BeautifulSoup(html_content, 'html.parser')

    if query == "text":
        return soup.get_text()

    elif query == "links":
        links = soup.find_all('a', href=True)
        return json.dumps([link['href'] for link in links])

    elif query == "metadata_and_headings":
        metadata = {
            "title": soup.title.string if soup.title else "",
            "meta_description": "",
            "headings": []
        }

        # メタディスクリプションを取得
        description_tag = soup.find('meta', attrs={"name": "description"})
        if description_tag:
            metadata["meta_description"] = description_tag.get("content", "")

        # ヘッディングを取得
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        metadata["headings"] = [heading.get_text() for heading in headings]

        return json.dumps(metadata)

    elif query == "html":
        return soup.prettify()

    elif query == "section_html":
        # 指定されたヘッディングの下のHTMLを収集する
        if heading:
            start_heading = soup.find(lambda tag: tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] and heading in tag.text)
            if not start_heading:
                return f"No section found with heading: {heading}"

            elements = []
            for sibling in start_heading.find_all_next():
                if sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    if sibling.name <= start_heading.name:
                        break
                elements.append(sibling)

            return ''.join([str(element) for element in elements]).strip()

        else:
            return "Heading must be specified for section_text query."

    else:
        return "Invalid query parameter"

