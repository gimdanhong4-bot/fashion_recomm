import os
import sys
from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import uvicorn
import json
from fastapi.responses import FileResponse

# SQLite3 버전 충돌 우회 (필수)
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import shutil
import torch
import re
import chromadb
from langchain_chroma import Chroma
import pandas as pd
import gradio as gr
import ast
import numpy as np
from PIL import Image
from collections import Counter, defaultdict
from transformers import CLIPProcessor, CLIPModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from sentence_transformers import util
from langchain_google_genai import ChatGoogleGenerativeAI

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime
import json

# [1] 구글 시트 연결 설정 함수
def get_google_sheet():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('fashion_key.json', scope)
    client = gspread.authorize(creds)
    sh = client.open('패션추천시스템_사용자정보')
    return sh

# [2] 검색 데이터 저장 함수 (시트1에 저장)
def log_search_to_sheet(bg, cat, min_p, max_p, style, txt, check, persona):
    try:
        sh = get_google_sheet()
        worksheet = sh.get_worksheet(0)

        if not worksheet.row_values(1):
            worksheet.append_row(['시간', '브랜드', '카테고리', '최소가격', '최대가격', '스타일', '텍스트', '성별교차', '결과페르소나'])

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        worksheet.append_row([timestamp, bg, cat, min_p, max_p, style, txt, "O" if check else "X", persona])
    except Exception as e:
        print(f"구글 시트 저장 실패: {e}")

# [3] 사이즈 추천 데이터 저장 함수 (시트2에 저장)
def log_size_to_sheet(sel_product, u_info, rec_size, reason):
    try:
        sh = get_google_sheet()
        try:
            worksheet = sh.get_worksheet(1)
        except:
            worksheet = sh.add_worksheet(title="사이즈추천", rows="100", cols="20")

        if not worksheet.row_values(1):
            worksheet.append_row(['시간', '선택상품', '유저정보', '추천사이즈', '추천이유'])

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_info_str = json.dumps(u_info, ensure_ascii=False)
        worksheet.append_row([timestamp, sel_product, user_info_str, rec_size, reason])
    except Exception as e:
        print(f"구글 시트 저장 실패: {e}")

device = "cuda" if torch.cuda.is_available() else "cpu"

base_path = "."
product_db_path = os.path.join(base_path, "product_db")
review_db_path = os.path.join(base_path, "review_db")
style_db_path = os.path.join(base_path, "style_db")

product_csv_path = os.path.join(base_path, "final_prod_data.csv")
size_csv_path = os.path.join(base_path, "size_sorted.csv")
review_csv_path = os.path.join(base_path, "final_review_df.csv")

try:
    prod_data = pd.read_csv(product_csv_path)
    prod_data['primary_id'] = prod_data['primary_id'].astype(str)
except:
    prod_data = pd.DataFrame()

try:
    prod_size_df = pd.read_csv(size_csv_path)
    prod_size_df = prod_size_df.replace('-', np.nan)
    prod_size_df['primary_id'] = prod_size_df['primary_id'].astype(str)
except:
    prod_size_df = pd.DataFrame()

try:
    review_df = pd.read_csv(review_csv_path)
    review_df['primary_id'] = review_df['primary_id'].astype(int)
    review_ids = set(review_df['primary_id'].astype(str))
except Exception as e: # 🚨 수정 완료: Exception as e 추가
    print(f"리뷰 데이터 로드 실패: {e}")
    review_ids = set()

clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

review_emb_func = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sroberta-multitask',
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.7)

try:
    product_client = chromadb.PersistentClient(path=product_db_path)
    product_collection = product_client.get_collection("product_vectorstore")
except Exception as e:
    print(f"{e} DB 연결 실패")

try:
    style_client = chromadb.PersistentClient(path=style_db_path)
    style_collection = style_client.get_collection("style_vectorstore")
except Exception as e:
    print(f"{e} DB 연결 실패")

try:
    review_db = Chroma(
        persist_directory=review_db_path,
        embedding_function=review_emb_func,
        collection_name="review_vectorstore"
    )
except Exception as e:
    print(f"{e} DB 연결 실패")

def get_style_options():
    try:
        data = style_collection.get(include=["metadatas"], limit=9999)
        names = [m.get("display_name") for m in data["metadatas"] if m.get("display_name")]
        names = sorted(set(names))
        return ["선택 안 함"] + names
    except Exception as e:
        print("ERROR in get_style_options:", e)
        return ["선택 안 함"]

style_key_options = get_style_options()

def set_fin_size(id, size_label):
    if prod_size_df.empty:
        return None
    try:
        r = prod_size_df[
            (prod_size_df['primary_id'] == str(id)) &
            (prod_size_df['size_cleaned'].astype(str) == str(size_label))
        ]
        if r.empty:
            return None
        numeric_cols = r.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ['primary_id']]
        return r.iloc[0][numeric_cols].to_dict()
    except:
        return None

def find_closest_size(target_id, ideal_spec):
    if prod_size_df.empty or not ideal_spec:
        return None
    target_r = prod_size_df[prod_size_df['primary_id'] == str(target_id)]
    if target_r.empty:
        return None
    if len(target_r) == 1:
        only_size = target_r.iloc[0]['size_cleaned']
        return only_size

    min_dist = float('inf')
    best_size = None
    common_keys = [k for k in ideal_spec.keys() if k in target_r.columns]

    if not common_keys:
        return None

    for _, row in target_r.iterrows():
        try:
            c_specs = [float(ideal_spec[k]) for k in common_keys if pd.notna(row[k])]
            t_specs = [float(row[k]) for k in common_keys if pd.notna(row[k])]
            if not c_specs or len(c_specs) != len(t_specs):
                continue
            dist = np.linalg.norm(np.array(c_specs) - np.array(t_specs))
            if dist < min_dist:
                min_dist = dist
                best_size = row['size_cleaned']
        except:
            continue
    return best_size

def search_recom_products(brand_lb, cat, sel_key, txt, img, user_info, min_price, max_price, check_option):
    query_vec, source = get_search_vector(sel_key, txt, img)
    if query_vec is None:
        return [], "검색 조건을 입력하세요", {}, user_info

    gender_option = [user_info['gender'], 'all']

    filter = {
      "$and": [
              {"brand_label": {"$eq": brand_lb}},
              {"sub_category": {"$eq": cat}}
              ]
    }

    if min_price is not None and max_price is not None:
        if min_price > max_price:
            min_price, max_price = max_price, min_price
        filter["$and"].append({"price": {"$gte": min_price}})
        filter["$and"].append({"price": {"$lte": max_price}})

    if not check_option:
        filter["$and"].append({"gender": {"$in": gender_option}})

    candidates = product_collection.query(
        query_embeddings=[query_vec],
        n_results=20,
        where=filter,
        include=["metadatas", "embeddings", "distances"]
    )

    if not candidates['ids'] or not candidates['ids'][0]:
        return [], "조건에 맞는 상품이 없습니다.", {}, user_info

    candidate_list_text = ""
    candidate_map = {}

    for i, meta in enumerate(candidates['metadatas'][0]):
        p_id = meta.get('primary_id')
        name = meta.get('product_name')
        brand = meta.get('brand')
        price = meta.get('price')
        dist = candidates['distances'][0][i]
        sim = 1 / (1 + dist)

        candidate_list_text += f"[{i+1}] ID: {p_id} | 브랜드: {brand} | 제품명: {name} | 스타일유사도: {sim:.4f}\n"
        candidate_map[str(i+1)] = {
            "id": p_id, "name": name, "brand": brand, "price": price,
            "cat": meta.get('sub_category'), "brand_lb": meta.get('brand_label'),
            "idx": i
        }

    style_requests = []
    if txt:
        style_requests.append(f"텍스트 요청: '{txt}'")
    if sel_key and sel_key != '선택 안 함':
        style_key_data = style_collection.get(
            where={"display_name": {"$eq": sel_key}},
            include=["metadatas"]
        )
        if style_key_data["metadatas"]:
            style_desc = style_key_data["metadatas"][0].get("description")
            if style_desc:
                style_requests.append(f"선택 스타일 키워드 설명: '{style_desc}'")

    if img is not None:
        style_requests.append("이미지 기반 스타일 검색 포함")

    user_input_str = ", ".join(style_requests) if style_requests else ""

    selection_persona_prompt = PromptTemplate.from_template("""
    당신은 전문 패션 MD입니다.
    사용자 프로필과 스타일 유사도 데이터를 바탕으로 선정된 후보 목록 20개를 분석하여, 가장 적합한 **제품 5개를 선정**하고 **패션 페르소나**를 정의하세요.
    [사용자 프로필]
    - 성별/체형: {g} / {h}cm / {w}kg
    - 선호 스타일 요청: "{s}"
    [후보 제품 목록 (유사도 순)]
    {c_list}
    [출력 형식 및 제약 사항]
    - 반드시 아래의 변수명을 포함하여 텍스트를 출력하세요.
    - 변수 양식 외에 다른 설명은 생략하세요.
    SELECTED_IDS=[순번1, 순번2, 순번3, 순번4, 순번5]
    PERSONA=여기에 3줄 내외의 페르소나 정의 작성
    """)

    # 에러 원천 차단: 무조건 변수방부터 만듭니다.
    selected_idx = []
    persona_text = "편안함과 개성을 추구하는 패션 스타일을 선호하는 페르소나입니다."

    try:
        response = llm.invoke(selection_persona_prompt.format(
            g=user_info.get('gender', 'male'), 
            h=user_info.get('height', 170), 
            w=user_info.get('weight', 65),
            s=user_input_str, 
            c_list=candidate_list_text
        ))
        
        # 1. 응답 데이터에서 순수 텍스트만 안전하게 추출
        raw_content = response.content
        if isinstance(raw_content, list):
            texts = [str(x.get('text', '')) if isinstance(x, dict) else str(x) for x in raw_content]
            selection_response = " ".join(texts)
        else:
            selection_response = str(raw_content)

        # 2. 🚨 핵심 추가: LangChain의 'extras' 메타데이터 꼬리가 붙어있다면 칼같이 절단!
        if "'extras'" in selection_response:
            selection_response = selection_response.split("'extras'")[0]
        if '"extras"' in selection_response:
            selection_response = selection_response.split('"extras"')[0]

        print("🤖 정제된 LLM 응답:\n", selection_response)

        # 3. ID 추출 로직
        text_for_ids = selection_response.split("PERSONA")[0] if "PERSONA" in selection_response else selection_response
        raw_numbers = re.findall(r'\d+', text_for_ids)
        for num in raw_numbers:
            if num in candidate_map and num not in selected_idx:
                selected_idx.append(num)
            if len(selected_idx) >= 5:
                break

        # 4. 페르소나 추출 로직
        extracted_p = ""
        if "PERSONA" in selection_response:
            extracted_p = selection_response.split("PERSONA")[1]
        elif "페르소나" in selection_response:
            extracted_p = selection_response.split("페르소나")[1]

        if extracted_p:
            # 등호나 콜론 등 쓸데없는 기호 제거
            extracted_p = extracted_p.replace("=", "").replace(":", "").replace("*", "").strip()
            
            # 🚨 괄호 { 나 [ 가 나타나면 시스템 메타데이터이므로 그 앞부분(순수 페르소나)만 남김
            persona_text = re.split(r'[{}[\]]', extracted_p)[0].strip()
            
            # 끝부분에 남은 따옴표나 쉼표 찌꺼기 완벽 청소
            persona_text = persona_text.strip("',\" \n")

    except Exception as e:
        print(f"페르소나 파싱 오류 발생: {e}")

    # 추출 실패 시 후보 1~5번 기본 선택
    if not selected_idx:
        selected_idx = list(candidate_map.keys())[:5]

    user_info['p'] = persona_text
    final_recom_item = []
    p_map = {}

    for idx_str in selected_idx[:5]:
        item_data = candidate_map[idx_str]
        img_url = get_img_url(item_data['id'])
        label = f"[{item_data['brand']}]_{item_data['name']} - {item_data['price']}원"

        final_recom_item.append((img_url, label))
        p_map[label] = item_data

    final_recom_item_gr = [[url if url else "", str(label)] for url, label in final_recom_item]

    return final_recom_item_gr, user_info['p'], p_map, user_info

def get_sim_products(target_id, category, brand_label=None, limit=5):
    target_data = product_collection.get(
        ids=[target_id], include=['embeddings', 'metadatas'])

    embs = target_data.get('embeddings')
    if embs is None or len(embs) == 0:
        return {}

    target_img = torch.tensor(embs[0], dtype=torch.float32)
    try:
        meta_size = target_data['metadatas'][0].get('size_vec', '[]')
        target_size = torch.tensor(ast.literal_eval(meta_size), dtype=torch.float32)
    except:
        return {}

    review_ids_list = list(review_ids)
    filter = {"$and": [{"sub_category": {"$eq": category}},
                       {"primary_id": {"$in": review_ids_list}},
                       {"review_count": { "$gt": 0 }}]}

    if brand_label is not None:
        filter["$and"].append({"brand_label": {"$eq": brand_label}})

    candidates = product_collection.get(
        where=filter, include=['embeddings', 'metadatas'])

    cand_ids = candidates.get('ids')
    if cand_ids is None or len(cand_ids) == 0:
        return {}

    valid_idx, candi_img, candi_size = [], [], []
    for i, p_id in enumerate(cand_ids):
        try:
            v_str = candidates['metadatas'][i].get('size_vec')
            if v_str:
                v = ast.literal_eval(v_str)
                if v:
                    candi_size.append(v)
                    candi_img.append(candidates['embeddings'][i])
                    valid_idx.append(i)
        except:
            continue

    if not valid_idx:
        return {}

    cand_img_vec = torch.tensor(np.array(candi_img), dtype=torch.float32)
    cand_size_vec = torch.tensor(np.array(candi_size), dtype=torch.float32)

    scores = (util.cos_sim(target_img, cand_img_vec)[0] * 0.6) + \
              (util.cos_sim(target_size, cand_size_vec)[0] * 0.4)

    top_k = torch.argsort(scores, descending=True)[:limit].tolist()

    result_map = {}
    for i in top_k:
        real_idx = valid_idx[i]
        p_id = cand_ids[real_idx]
        score = float(scores[i])
        result_map[p_id] = score

    return result_map

def recom_size(label, p_map, user_info):
    if not label:
        return "상품을 먼저 선택해주세요", ""
    if label not in p_map:
        return f"오류: 선택한 제품 '{label}'의 데이터를 찾을 수 없습니다.", ""

    user_info = dict(user_info) if user_info else {}
    target_id, cat, bl = p_map[label]['id'], p_map[label]['cat'], p_map[label]['brand_lb']

    hyde_prompt = PromptTemplate.from_template("""
당신은 지금부터 아래 프로필을 가진 사용자가 되어, 방금 구매한 옷에 대해 **만족한 후기**를 작성해야 합니다.
[사용자 프로필]
- 성별: {g}
- 신체 스펙: {h}cm / {w}kg
- 사용자 페르소나: {p}
[구매한 상품 정보]
- 제품 기본 정보: {p_info}
- 카테고리: {cat} 
[작성 지침]
1. 당신의 신체 스펙과 선호 스타일을 고려하여 사용자가 제품 핏에 만족한다는 가정 하에 리뷰를 작성하세요.
2. **주의:** S, M, L 같은 **구체적인 사이즈(옵션명)는 절대 언급하지 마세요.** 오직 '핏감(Fit)'과 '착용감'에 대해서만 이야기하세요.
3. 리뷰는 구매한 상품 정보에 기반하여 작성하세요.
4. 말투는 자연스러운 한국어 구어체(리뷰톤)를 사용하며 3줄 이내로 작성하세요.
""")

    # 🚨 리스트 에러 근본 해결을 위한 예외처리 래핑
    try:
        hyde_response = llm.invoke(hyde_prompt.format(
            h=user_info.get('height','175'), w=user_info.get('weight','70'),
            g=user_info.get('gender', 'male'), p=user_info.get('p','기본 페르소나'),
            p_info=label, cat=cat))
        
        hyde_txt = hyde_response.content
        if isinstance(hyde_txt, list):
            hyde_txt = " ".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in hyde_txt])
        hyde_txt = str(hyde_txt)
    except Exception as e:
        print(f"HyDE 생성 에러: {e}")
        hyde_txt = "사이즈가 아주 잘 맞고 편안합니다. 실루엣이 마음에 듭니다."

    hyde_vec = review_emb_func.embed_query(hyde_txt)

    product_score_map = {str(target_id): 1.0}
    search_pool_ids = [str(target_id)]
    method_log = f"해당 추천 상품"

    sim_ids_2 = get_sim_products(target_id, cat, brand_label=bl, limit=5)
    if sim_ids_2:
        product_score_map.update(sim_ids_2)
        search_pool_ids.extend(sim_ids_2.keys())
        method_log += f" + 같은 브랜드 그룹 내 유사도 높은 상품 {len(sim_ids_2)}개"

    if len(search_pool_ids) < 3:
        sim_ids_3 = get_sim_products(target_id, cat, brand_label=None, limit=5)
        for id, score in sim_ids_3.items():
            if id not in product_score_map:
                product_score_map[id] = score
                search_pool_ids.append(id)
        method_log += f" + 같은 카테고리 내 유사도 높은 상품 {len(sim_ids_3)}개"

    gender = user_info.get('gender', 'male')
    height = int(user_info.get('height', 175))
    weight = int(user_info.get('weight', 70))

    min_h, max_h = height - 3, height + 3
    min_w, max_w = weight - 3, weight + 3

    final_docs_raw = review_db.similarity_search_by_vector(
        hyde_vec, k=30,
        filter={
            "$and": [
                {"product_id": {"$in": search_pool_ids}},
                {"gender": {"$eq": gender}},
                {"height": {"$gte": min_h}},
                {"height": {"$lte": max_h}},
                {"weight": {"$gte": min_w}},
                {"weight": {"$lte": max_w}}
            ]
        })

    if not final_docs_raw:
        relaxed_filter = {"$and": [{"product_id": {"$in": search_pool_ids}}, {"gender": {"$eq": gender}}]}
        final_docs_raw = review_db.similarity_search_by_vector(hyde_vec, k=30, filter=relaxed_filter)

    if not final_docs_raw:
        return "유사 리뷰 데이터 부족", method_log

    best_doc, max_score, weight_docs = None, -1e9 , []
    size_score = defaultdict(float)
    size_count = defaultdict(int)
    valid_doc_cnt = 0

    for doc in final_docs_raw:
        meta = getattr(doc, 'metadata', {}) or {}
        size = meta.get('size')
        if size:
            size_count[size] += 1
            valid_doc_cnt += 1

    if valid_doc_cnt == 0:
        return "best_doc is not found", method_log

    for doc in final_docs_raw:
        meta = getattr(doc, 'metadata', {}) or {}
        prod_id = meta.get('product_id')
        size = meta.get('size')

        if not prod_id or not size:
            continue

        dis = meta.get("score", 0.0)
        review_sim = max(0.0, 1.0 - dis)
        s_cnt = size_count[size]
        size_ratio_score =(1 + s_cnt / valid_doc_cnt)

        prod_id = str(prod_id)
        prod_sim = float(product_score_map.get(prod_id, 0.5))
        w_score = prod_sim * review_sim * size_ratio_score

        weight_docs.append({"doc": doc, "id": prod_id, "weight": w_score, "size": size})
        size_score[size] += w_score

        if w_score > max_score:
            max_score = w_score
            best_doc = {"product_id": prod_id, "size": size}

    if not best_doc:
        return "best_doc is not found", method_log

    ref_p, ref_sz = best_doc['product_id'], best_doc['size']
    ideal_spec = set_fin_size(ref_p, ref_sz)
    final_sz, info = ref_sz, "(유사 제품 사이즈 추천)"

    target_r = prod_size_df[prod_size_df['primary_id'] == str(target_id)]
    size_cnt = len(target_r)

    if size_cnt == 1:
        final_sz = target_r.iloc[0]['size_cleaned']
        info = "(원사이즈 제품)"
    elif ideal_spec:
        actual_sz = find_closest_size(target_id, ideal_spec)
        if actual_sz:
            final_sz = actual_sz
            info = f"(유사 제품 {ref_p} [{ref_sz}] 실측 기반 매핑)"

    weight_docs.sort(key=lambda x: x['weight'], reverse=True)
    evidence_txt = '\n'.join([f"-[{doc['size']}] {doc['doc'].page_content[:80]}...({doc['weight']:.2f})" for doc in weight_docs[:5]])

    prompt = "패션 상품 구매를 고려하고 있는 사용자에게 '{s}' 사이즈 추천. 근거:\n{e}\n페르소나: {p}\n(로직: {m}, {i})"
    
    # 🚨 리스트 에러 안전 변환 가드 추가
    msg_response = llm.invoke(prompt.format(
        s=final_sz, e=evidence_txt, p=user_info.get('p',''), m=method_log, i=info))
    msg = msg_response.content
    if isinstance(msg, list):
        msg = " ".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in msg])
    msg = str(msg)

    return final_sz, msg

def get_img_url(id):
    if prod_data.empty:
        return None
    try:
        r = prod_data[prod_data['primary_id'] == str(id).strip()]
        if not r.empty:
            url = r.iloc[0]['thumbnail_img_url']
            if pd.notna(url):
                return str(url)
    except:
        pass
    return None

def get_search_vector(selected_key, user_txt, img_path):
    vector_list = []
    message_list = []

    if img_path:
        image = Image.open(img_path)
        inputs = clip_processor(images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            img_vec = clip_model.get_image_features(**inputs).cpu().numpy().tolist()[0]
        vector_list.append(img_vec)
        message_list.append("이미지 기반 검색")

    if user_txt:
        inputs = clip_processor(text=[user_txt], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            txt_vec = clip_model.get_text_features(**inputs).cpu().numpy().tolist()[0]
        vector_list.append(txt_vec)
        message_list.append(f"{user_txt}에 대한 검색 중...")

    if selected_key and selected_key != '선택 안 함':
        items = style_collection.get(include=["metadatas", "embeddings"])
        for m, e in zip(items["metadatas"], items["embeddings"]):
            if m.get("display_name") == selected_key:
                vector_list.append(e)
                message_list.append(f"{selected_key}에 대한 검색 중...")
                break

    if not vector_list:
        return None, "검색 조건을 입력하세요"
    else:
      mean_vector = torch.mean(torch.tensor(vector_list), dim=0).tolist()
      return mean_vector, ", ".join(message_list)

def on_gallery_click(p_map, evt: gr.SelectData):
    if not p_map:
        return ""
    keys_list = list(p_map.keys())
    if evt.index < len(keys_list):
        selected_label = keys_list[evt.index]
        return selected_label
    return ""

def update_cat(brand_label):
    if not brand_label:
        return gr.Dropdown(choices=[], value=None)
    try:
        filtered_df = prod_data[prod_data['brand_label'] == brand_label]
        new_choices = sorted(filtered_df['category_final'].dropna().unique().tolist())
        new_value = new_choices[0] if new_choices else None
        return gr.Dropdown(choices=new_choices, value=new_value, label="카테고리 (서브)")
    except Exception as e:
        print(f" 카테고리 업데이트 실패: {e}")
        return gr.Dropdown(choices=[], value=None)

def update_price(brand_label, cat):
    min_price = int(prod_data['sale_price'].dropna().min())
    max_price = int(prod_data['sale_price'].dropna().max())

    if not brand_label or not cat:
        return (f"가격범위: {min_price:,}원 ~ {max_price:,}원", min_price, max_price, gr.update(visible=False))

    try:
        selected_prod_df = prod_data[(prod_data['brand_label'] == brand_label) & (prod_data['category_final'] == cat)]
        update_min_price = int(selected_prod_df['sale_price'].min())
        update_max_price = int(selected_prod_df['sale_price'].max())
        price_text_info = f"가격범위: {update_min_price:,}원 ~ {update_max_price:,}원"
        return price_text_info, update_min_price, update_max_price, gr.update(visible=False)
    except Exception as e:
        print(f"가격 업데이트 실패: {e}")
        return min_price, max_price

brand_description = {
    "미니멀리즘, 높은 퀄리티, 편안함, 타임리스, 웨어러블": "미니멀하고 타임리스한 디자인을 통해 높은 품질의 편안한 의류를 제공하며, 실용성과 웨어러블함을 중요시합니다.",
    "편안함, 스타일, 개성, 실용성, 자유로움": "이 클러스터의 브랜드들은 편안한 실루엣과 실용적인 디자인을 통해 개인의 개성을 표현하며, 일상에서 자유로운 스타일을 추구합니다.",
    "편안함, 일상, 유연함, 트렌디, 개인성": "일상 속에서 편안하게 착용할 수 있는 트렌디한 디자인을 추구하며, 개인의 개성과 다양한 라이프스타일을 반영하는 브랜드들이 모여 있습니다.",
    "편안함, 유니크함, 지속 가능성, 반항적 디자인, 일상성과 특별함": "이 클러스터의 브랜드들은 일상 속에서 편안함과 유니크함을 추구하며, 반항적이고 독창적인 디자인으로 새로운 흐름을 제안하는 동시에 지속 가능한 삶을 지향합니다.",
    "편안함, 개성, 자연스러움, 균형, 다채로움": "이 클러스터의 브랜드들은 편안함과 개성을 중시하며, 자연스러운 스타일과 균형 잡힌 라이프스타일을 통해 다양성을 존중하는 독창적인 패션을 제안합니다.",
    "아웃도어, 기능성, 일상, 스포츠, 혁신": "스포츠와 아웃도어 활동을 일상 속에서 즐길 수 있도록 기능성과 스타일을 겸비한 다양한 제품을 제공하는 브랜드들이 모인 클러스터입니다.",
    "여성, 클래식, 트렌디, 웨어러블, 심플, 감성": "이 클러스터의 브랜드들은 클래식함과 트렌디함을 조화롭게 담아내며, 편안한 착용감을 제공하는 웨어러블한 디자인과 감성적인 스타일을 강조합니다."
}

def load_stye_collection():
    try:
        data = style_collection.get(include=["metadatas"], limit=9999)
        desc_dict = {}
        for meta in data['metadatas']:
            style_key = meta.get('display_name')
            style_desc = meta.get('description')
            if style_key and style_desc and style_key not in desc_dict:
                desc_dict[style_key] = style_desc
        return desc_dict
    except:
        return {}

style_description = load_stye_collection()

brand_label_choice, subcat_choice = [], []
min_price, max_price = 0, 100

if not prod_data.empty:
    brand_label_choice = sorted(prod_data['brand_label'].dropna().unique().tolist())
    subcat_choice = sorted(prod_data['category_final'].dropna().unique().tolist())
    min_price = int(prod_data['sale_price'].dropna().min())
    max_price = int(prod_data['sale_price'].dropna().max())

def get_brand_desc(brand_label):
    return brand_description.get(brand_label, "")

def get_style_desc(style):
    return style_description.get(style, "")

def format_info_box(text):
    if not text:
        return ""
    return f"""
    <div style="
        background-color: #f0f9ff;
        color: #333333;
        border: 1px solid #bae6fd;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 0.95em;
        margin-top: 5px;
    ">
        {text}
    </div>
    """

app = FastAPI()

@app.get("/")
async def read_root():
    return FileResponse("new_index.html")

# CORS 설정 (HTML과 통신하기 위해 필수)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# [API 1] 검색 및 추천 (기존 search_and_log 역할)
@app.post("/api/search")
async def api_search(
    bg: str = Form(...),
    cat: str = Form(...),
    sl: str = Form("선택 안 함"),
    tx: str = Form(""),
    g: str = Form(...),
    h: float = Form(...),
    w: float = Form(...),
    min_p: int = Form(...),
    max_p: int = Form(...),
    check: bool = Form(False),
    img: UploadFile = File(None)
):
    try:
        # 1. 이미지 임시 저장 (이미지가 있는 경우)
        img_path = None
        if img and img.filename:
            img_path = f"temp_{img.filename}"
            with open(img_path, "wb") as buffer:
                shutil.copyfileobj(img.file, buffer)

        # 2. 유저 정보 구성
        user_info = {'gender': g, 'height': h, 'weight': w}

        # 3. 기존 검색 로직 호출
        results = search_recom_products(bg, cat, sl, tx, img_path, user_info, min_p, max_p, check)
        
        gal_data, persona, p_map, updated_user_info = results

        # 4. 로그 저장
        log_search_to_sheet(bg, cat, min_p, max_p, sl, tx, check, persona)

        # 5. 사용된 임시 이미지 삭제
        if img_path and os.path.exists(img_path):
            os.remove(img_path)

        return {
            "success": True,
            "gallery": gal_data,
            "persona": persona,
            "p_map": p_map,
            "user_info": updated_user_info
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# 사이즈 추천용 데이터 모델
class SizeRecommendRequest(BaseModel):
    sel_product: str
    p_map: dict
    user_info: dict

# [API 2] 사이즈 추천 (기존 size_recom_and_log 역할)
@app.post("/api/recommend_size")
async def api_recommend_size(data: SizeRecommendRequest):
    try:
        rec_size, reason = recom_size(data.sel_product, data.p_map, data.user_info)
        log_size_to_sheet(data.sel_product, data.user_info, rec_size, reason)
        
        return {
            "success": True,
            "rec_size": rec_size,
            "reason": reason
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)