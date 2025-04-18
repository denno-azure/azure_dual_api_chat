import requests
import tempfile
import pdfplumber
import hashlib
import base64
from PyPDF2 import PdfReader
import re
import os

pdf = {
    "type": "function",
    "function": {
        "name": "extract_pdf_content",
        "description": "与えられたPDFから、全体または指定範囲・画像ID単位で構造化情報（目次・テキスト・画像情報など）を抽出します。注: 現時点ではOCRを行わないため、スキャン画像だけのPDFはテキストを取得できません。",
        "parameters": {
          "type": "object",
          "properties": {
            "pdf_url": {
              "type": "string",
              "description": "解析対象PDFの公開URL（OCRは行いません）"
            },
            "page_range": {
              "type": "string",
              "description": "抽出対象のページ範囲（例：'1-5', '2,4,6-8' ）。未指定の場合は自動分岐。"
            },
            "image_id": {
              "type": "string",
              "description": "抽出したい画像のID。画像IDはまずpdf_urlのみ指定して得られるJSONのimagesに含まれます。"
            }
          },
          "required": ["pdf_url"]
        }
    }
}

def get_pdf_from_url(pdf_url):
    """
    PDFをダウンロードし、バイト列を返す。
    失敗時にはExceptionをthrowするので呼び出し元でキャッチ。
    """
    response = requests.get(pdf_url)
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve PDF: status={response.status_code}")
    return response.content

def get_file_hash(pdf_bytes):
    """
    PDFファイルのSHA256ハッシュ（先頭8文字）を返してID生成に使う。
    大きなファイルは想定外だが、将来的にはストリームハッシュなど要検討。
    """
    return hashlib.sha256(pdf_bytes).hexdigest()[:8]

def parse_page_range(page_range_str, max_pages):
    """
    '1-3','5,7-9'などの文字列をパースし、対象ページIndexの配列を返す。
    PDFは0-indexでアクセスするため注意。
    """
    if not page_range_str:
        return None
    pages = set()
    for part in page_range_str.split(','):
        if '-' in part:
            start, end = part.split('-')
            pages.update(range(int(start) - 1, int(end)))
        else:
            pages.add(int(part) - 1)
    # 範囲外を排除しソート
    return [p for p in sorted(pages) if 0 <= p < max_pages]

# ──────────────────────────────────────────────────────────
# 見出し・段落抽出用（簡易バージョン）
# 将来的には、ページから取得できる char 情報やフォントサイズ等を
# 利用して精緻化できるよう、引数に char_data などを追加しておく。
# ──────────────────────────────────────────────────────────
def extract_headings_and_paragraphs(text, char_data=None):
    """
    text: ページから抽出した生テキスト
    char_data: 将来的に文字単位フォントサイズ等が格納される想定
               (今は未使用だが、拡張しやすいよう引数だけ用意)
    return: [
      { "heading": <見出し行 or None>, "paragraphs": [<string>...] },
      ...
    ]
    """
    lines = text.split('\n') if text else []
    # 簡易的に「行頭が数字や章などを示す文」を見出しとみなす正規表現例
    heading_pat = re.compile(r'^(第?\d+[\.\-章節部]|[A-Z][\.\d ]+|[0-9]+\.)')

    result = []
    cur_section = {"heading": None, "paragraphs": []}

    for line in lines:
        lstr = line.strip()
        # 見出しパターンに合致したら、新しいセクションを開始
        if heading_pat.match(lstr):
            # 一つ前のセクションを追加
            if cur_section["heading"] or cur_section["paragraphs"]:
                result.append(cur_section)
            cur_section = {"heading": lstr, "paragraphs": []}
        elif lstr:
            # 行が空白でなければ、段落として貯める
            cur_section["paragraphs"].append(lstr)

    # 最後のセクションが残れば追加
    if cur_section["heading"] or cur_section["paragraphs"]:
        result.append(cur_section)

    """
    ここで char_data があれば、見出し候補の行のフォントサイズなどを解析→
    より正確な見出し判定が可能。 -> 今は未実装。
    """
    return result

def extract_bookmarks(pdf_path):
    """
    PyPDF2でブックマーク(アウトライン)を取得。
    Multi-levelの場合はlevelを深掘りして再帰的に構造を返す。
    """
    try:
        # Readerのバージョンによっては .outline でなく get_outlines() など使う場合も
        reader = PdfReader(pdf_path)
        outlines = getattr(reader, "outline", None)
        if outlines is None:
            # 旧バージョンだと get_outlines() か getOutlines()
            outlines = reader.get_outlines()
    except Exception:
        return []

    def _parse_outline(items, depth=1):
        for o in items:
            if isinstance(o, list):
                yield from _parse_outline(o, depth+1)
            else:
                try:
                    yield {
                        "title": o.title,
                        "page_number": reader.get_destination_page_number(o) + 1,
                        "level": depth
                    }
                except:
                    # 取得できないものはスキップ
                    pass

    # outlines が存在しない、空などの場合は空リスト
    if not outlines:
        return []
    return list(_parse_outline(outlines, 1))

def get_image_summaries(page, page_num, file_hash):
    """
    pdfplumberのpage.imagesで得られる情報から、画像のID・座標情報などを作る。
    将来的には実際の画像バイトを取り、高度なID生成を行うとより確実。
    """
    imgs = []
    for img in page.images:
        bbox = (img['x0'], img['top'], img['x1'], img['bottom'])
        # pdfplumberの'img["name"]'はXObject名で一意性が絶対ではないが、簡易実装
        img_id = f"{file_hash}_p{page_num}_{img['name']}"
        imgs.append({
            "image_id": img_id, 
            "page": page_num, 
            "bbox": bbox
        })
    return imgs

def extract_images_for_id(pdf_path, image_id, file_hash):
    """
    image_id をもとに該当画像を探し、Base64化して返却する。(簡易実装)
    OCRは行わない。
    """
    import io

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                for img in page.images:
                    candidate_id = f"{file_hash}_p{i+1}_{img['name']}"
                    if candidate_id == image_id:
                        try:
                            # バウンディングボックスを使って画像を切り抜き
                            cropbox = (img['x0'], img['top'], img['x1'], img['bottom'])
                            cropped_page = page.within_bbox(cropbox)
                            # さらに画像抽出
                            if cropped_page.images:
                                pil_img = cropped_page.to_image(resolution=150).original.convert("RGB")
                                buf = io.BytesIO()
                                pil_img.save(buf, format="PNG")
                                buf.seek(0)
                                b64data = base64.b64encode(buf.read()).decode("utf-8")
                                return {
                                    "type": "image",
                                    "status": "success",
                                    "image_id": image_id,
                                    "image_base64": b64data,
                                    "mime_type": "image/png"
                                }
                        except Exception as e:
                            return {
                                "type": "image",
                                "status": "error",
                                "reason": f"Image extraction failed: {str(e)}",
                                "image_id": image_id
                            }
    except Exception as ex:
        return {
            "type": "image",
            "status": "error",
            "reason": f"Failed to open PDF in image extraction: {str(ex)}",
            "image_id": image_id
        }

    # 該当画像が見つからない場合
    return {"type": "image", "status": "not_found", "reason": "Image ID not found", "image_id": image_id}

def extract_pdf_content(pdf_url, page_range=None, image_id=None):
    """
    PDFのテキスト・画像・アウトラインをfunction-calling向けJSONで返すメイン関数。
    【注意】この関数はOCRを行わないため、スキャン画像のみのPDFではテキストは抽出されません。
    """
    meta = {"status": "success"}

    # PDFのダウンロード
    try:
        pdf_bytes = get_pdf_from_url(pdf_url)
    except Exception as e:
        return {"status": "error", "reason": str(e)}

    file_hash = get_file_hash(pdf_bytes)

    # 一時ファイルに保存してPyPDF2/pdfplumberでオープン
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpf:
        tmpf.write(pdf_bytes)
        tmpf.flush()
        pdf_path = tmpf.name

    # if image_id 指定 → 画像抽出のみ
    if image_id:
        res = extract_images_for_id(pdf_path, image_id, file_hash)
        # 後始末
        try:
            os.remove(pdf_path)
        except:
            pass
        return res

    # ブックマーク(アウトライン)やページ数判定等
    try:
        reader = PdfReader(pdf_path)
        n_pages = len(reader.pages)
    except Exception as e:
        # 後始末
        try:
            os.remove(pdf_path)
        except:
            pass
        return {"status": "error", "reason": f"Failed to read PDF metadata: {str(e)}"}

    try:
        if page_range:
            # ページ範囲指定された場合
            pages_idx = parse_page_range(page_range, n_pages)
            extracted_pages = []
            images = []
            with pdfplumber.open(pdf_path) as pdf:
                for idx in pages_idx:
                    page = pdf.pages[idx]
                    text = page.extract_text()
                    # 将来 char_data = page.chars or something
                    # ここでフォントサイズ情報解析を掛けて引数に渡すことも可能
                    sections = extract_headings_and_paragraphs(text, char_data=None)
                    extracted_pages.append({
                        "page": idx + 1,
                        "sections": sections
                    })
                    images += get_image_summaries(page, idx+1, file_hash)

            # 後始末
            os.remove(pdf_path)
            return {
                "type": "partial",
                "pages": extracted_pages,
                "images": images,
                "page_range": page_range,
                **meta
            }

        if n_pages <= 20:
            # 全ページ抽出
            extracted_pages = []
            images = []
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    sections = extract_headings_and_paragraphs(text, char_data=None)
                    extracted_pages.append({
                        "page": i + 1,
                        "sections": sections
                    })
                    images += get_image_summaries(page, i+1, file_hash)

            os.remove(pdf_path)
            return {
                "type": "full",
                "all_pages_extracted": True,
                "pages": extracted_pages,
                "images": images,
                **meta
            }
        else:
            # 21ページ超の場合、まずブックマーク(アウトライン)を試みる
            outline = extract_bookmarks(pdf_path)
            if outline:
                # 後始末
                os.remove(pdf_path)
                return {
                    "type": "outline",
                    "outline_available": True,
                    "all_pages_extracted": False,
                    "outline": outline,
                    **meta
                }
            else:
                # 目次が無ければ冒頭20ページのみ抽出
                extracted_pages = []
                images = []
                with pdfplumber.open(pdf_path) as pdf:
                    limit = min(20, n_pages)
                    for i in range(limit):
                        page = pdf.pages[i]
                        text = page.extract_text()
                        sections = extract_headings_and_paragraphs(text, char_data=None)
                        extracted_pages.append({
                            "page": i + 1,
                            "sections": sections
                        })
                        images += get_image_summaries(page, i+1, file_hash)

                os.remove(pdf_path)
                return {
                    "type": "partial_preview",
                    "all_pages_extracted": False,
                    "pages": extracted_pages,
                    "images": images,
                    "partial_extraction_reason": "NoOutlineFound_21+pages",
                    **meta
                }

    except Exception as e:
        # 途中で何か失敗したら
        try:
            os.remove(pdf_path)
        except:
            pass
        return {"status": "error", "reason": f"Processing failed: {str(e)}"}
    finally:
        # 正常終了もしくは例外でもファイル削除（念のため）
        if os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except:
                pass
