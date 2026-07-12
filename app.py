import os
import sys
import shutil
import tempfile
import json
import datetime
import re
import ast
import numpy as np
import pandas as pd
import torch
from PIL import Image
from collections import Counter, defaultdict

# Streamlit 패키지
import streamlit as st

# ChromaDB SQLite3 버전 충돌 해결용 우회 코드 (Streamlit Cloud 배포 시 필수)
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from langchain_chroma import Chroma
from transformers import CLIPProcessor, CLIPModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from sentence_transformers import util
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# 🚀 OpenAI 대신 Gemini(Google GenAI) 라이브러리 가져오기
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========================================
# [1] API 및 기본 설정
# ==========================================
key_content = os.environ.get("GSPREAD_KEY")

if key_content:
    with open("fashion_key.json", "w") as f:
        f.write(key_content)
    print("✅ 보안 키 파일 생성 완료!")
else:
    print("⚠️ 경고: GSPREAD_KEY 환경변수가 없습니다. (로컬/Streamlit Cloud Secrets 확인)")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# [2] 파일 경로 및 DB 복사 로직 설정
# ==========================================
base_path = "."

# GitHub 등에 바로 올릴 경우를 대비해 drive 경로와 local 경로를 동일한 베이스로 매핑
drive_product_db_path = os.path.join(base_path, "product_db")
drive_review_db_path = os.path.join(base_path, "review_db")
drive_style_db_path = os.path.join(base_path, "style_db")

local_product_db_path = os.path.join(base_path, "local_product_db")
local_review_db_path = os.path.join(base_path, "local_review_db")
local_style_db_path = os.path.join(base_path, "local_style_db")

product_csv_path = os.path.join(base_path, "final_prod_data.csv")
size_csv_path = os.path.join(base_path, "size_sorted.csv")
review_csv_path = os.path.join(base_path, "final_review_df.csv")

def copy_db_to_local(db_path, local_path):
    if os.path.exists(db_path) and os.listdir(db_path):
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        shutil.copytree(db_path, local_path)
        return True
    else:
        return False

# DB 복사 시도 (Streamlit Cloud에서는 원본 폴더를 그대로 사용하도록 예외처리)
db_ready = True
if not copy_db_to_local(drive_product_db_path, local_product_db_path):
    local_product_db_path = drive_product_db_path  # 복사 실패 시 원본 경로 사용
if not copy_db_to_local(drive_review_db_path, local_review_db_path):
    local_review_db_path = drive_review_db_path
if not copy_db_to_local(drive_style_db_path, local_style_db_path):
    local_style_db_path = drive_style_db_path

# ==========================================
# [3] 데이터 및 모델, DB 로드
# ==========================================
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
except:
    review_ids = set()

# 모델 로드
clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

review_emb_func = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sroberta-multitask',
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)

# 🚀 LLM을 Gemini 모델로 변경
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# DB 연결
try:
    product_client = chromadb.PersistentClient(path=local_product_db_path)
    product_collection = product_client.get_collection("product_vectorstore")
except Exception as e:
    print(f"Product DB 연결 실패: {e}")

try:
    style_client = chromadb.PersistentClient(path=local_style_db_path)
    style_collection = style_client.get_collection("style_vectorstore")
except Exception as e:
    print(f"Style DB 연결 실패: {e}")

try:
    review_db = Chroma(
        persist_directory=local_review_db_path,
        embedding_function=review_emb_func,
        collection_name="review_vectorstore"
    )
except Exception as e:
    print(f"Review DB 연결 실패: {e}")

# ==========================================
# [4] 백엔드 로직 함수
# ==========================================
def get_google_sheet():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('fashion_key.json', scope)
    client = gspread.authorize(creds)
    sh = client.open('패션추천시스템_사용자정보')
    return sh

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

def get_style_options():
    try:
        data = style_collection.get(include=["metadatas"], limit=9999)
        names = [m.get("display_name") for m in data["metadatas"] if m.get("display_name")]
        names = sorted(set(names))
        return ["선택 안 함"] + names
    except Exception as e:
        return ["선택 안 함"]

style_key_options = get_style_options()

def set_fin_size(id, size_label):
    if prod_size_df.empty: return None
    try:
        r = prod_size_df[
            (prod_size_df['primary_id'] == str(id)) &
            (prod_size_df['size_cleaned'].astype(str) == str(size_label))
        ]
        if r.empty: return None
        numeric_cols = r.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ['primary_id']]
        return r.iloc[0][numeric_cols].to_dict()
    except:
        return None

def find_closest_size(target_id, ideal_spec):
    if prod_size_df.empty or not ideal_spec: return None
    target_r = prod_size_df[prod_size_df['primary_id'] == str(target_id)]
    if target_r.empty: return None
    if len(target_r) == 1:
        return target_r.iloc[0]['size_cleaned']

    min_dist = float('inf')
    best_size = None
    common_keys = [k for k in ideal_spec.keys() if k in target_r.columns]
    if not common_keys: return None

    for _, row in target_r.iterrows():
        try:
            c_specs = [float(ideal_spec[k]) for k in common_keys if pd.notna(row[k])]
            t_specs = [float(row[k]) for k in common_keys if pd.notna(row[k])]
            if not c_specs or len(c_specs) != len(t_specs): continue
            dist = np.linalg.norm(np.array(c_specs) - np.array(t_specs))
            if dist < min_dist:
                min_dist = dist
                best_size = row['size_cleaned']
        except:
            continue
    return best_size

def get_img_url(id):
    if prod_data.empty: return None
    try:
        r = prod_data[prod_data['primary_id'] == str(id).strip()]
        if not r.empty:
            url = r.iloc[0]['thumbnail_img_url']
            if pd.notna(url): return str(url)
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

def search_recom_products(brand_lb, cat, sel_key, txt, img, user_info, min_price, max_price, check_option):
    query_vec, source = get_search_vector(sel_key, txt, img)
    if query_vec is None: return [], "검색 조건을 입력하세요", {}, "", []

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

    if not candidates['ids'][0]:
        return [], "조건에 맞는 상품이 없습니다.", {}, "", []

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
    if txt: style_requests.append(f"텍스트 요청: '{txt}'")
    if sel_key and sel_key != '선택 안 함':
        style_key_data = style_collection.get(where={"display_name": {"$eq": sel_key}}, include=["metadatas"])
        if style_key_data["metadatas"]:
            style_desc = style_key_data["metadatas"][0].get("description")
            if style_desc: style_requests.append(f"선택 스타일 키워드 설명: '{style_desc}'")
    if img is not None: style_requests.append("이미지 기반 스타일 검색 포함")

    user_input_str = ", ".join(style_requests) if style_requests else ""

    selection_persona_prompt = PromptTemplate.from_template("""
    당신은 전문 패션 MD입니다.
    사용자 프로필과 스타일 유사도 데이터를 바탕으로 선정된 후보 목록 20개를 분석하여, 가장 적합한 **제품 5개를 선정**하고 **패션 페르소나**를 정의하세요.

    [사용자 프로필]
    - 성별/체형: {g} / {h}cm / {w}kg
    - 선호 스타일 요청: "{s}"

    [후보 제품 목록 (유사도 순)]
    {c_list}

    [사고 과정 (Chain of thought)]
    1. 사용자 분석: 사용자 프로필 정보를 고려하여 사용자의 성별/체형에서 사용자의 선호 스타일을 표현하기 위해서는 어떤 실루엣, 소재, 디자인이 가장 적합한지를 기준을 세우세요.
    2. 추천 제품 선정: 위에서 세운 기준에 근거하여 사용자에게 가장 적합한 제품 5개를 선정하세요.
    3. 페르소나 도출: 선별된 5개 제품과 사용자의 특징을 종합하여, 이 사용자가 추구하는 패션 아이덴티티(페르소나)를 구체적인 코디 예시를 들어 묘사합니다.

    [출력 형식 및 제약 사]
    1. 사고 과정(설명)은 출력하지 말고, 오직 **결과 데이터**만 아래 형식으로 출력하세요.
    2. 제품번호는 후보 목록에 표시된 **순번(숫자)**만 적으세요.

    SELECTED_IDS = [순번1, 순번2, 순번3, 순번4, 순번5]
    |||
    PERSONA = (여기에 3줄 내외의 페르소나 정의 작성)
    """)

    selection_response = llm.invoke(selection_persona_prompt.format(
        g=user_info['gender'], h=user_info['height'], w=user_info['weight'],
        s=user_input_str, c_list=candidate_list_text
    )).content

    try:
        parts = selection_response.split("|||")
        ids_text = parts[0]
        selected_idx = re.findall(r'\d+', ids_text)
        selected_idx = [idx for idx in selected_idx if idx in candidate_map]
        persona_text = parts[1].replace("PERSONA =", "").strip()
    except Exception as e:
        selected_idx = list(candidate_map.keys())[:5]
        persona_text = "페르소나를 도출하지 못했습니다."

    if not selected_idx:
        selected_idx = list(candidate_map.keys())[:5]

    user_info['p'] = persona_text
    final_recom_item = []
    p_map = {}

    for idx_str in selected_idx[:5]:
        item_data = candidate_map[idx_str]
        img_url = get_img_url(item_data['id'])
        label = f"[{item_data['brand']}]_{item_data['name']}] - {item_data['price']}원"
        final_recom_item.append((img_url, label))
        p_map[label] = item_data

    final_recom_item_gr = [[url if url else "", str(label)] for url, label in final_recom_item]
    return final_recom_item_gr, user_info['p'], p_map, user_info

def get_sim_products(target_id, category, brand_label=None, limit=5):
    target_data = product_collection.get(ids=[target_id], include=['embeddings', 'metadatas'])
    embs = target_data.get('embeddings')
    if embs is None or len(embs) == 0: return {}

    target_img = torch.tensor(embs[0], dtype=torch.float32)
    try:
        meta_size = target_data['metadatas'][0].get('size_vec', '[]')
        target_size = torch.tensor(ast.literal_eval(meta_size), dtype=torch.float32)
    except:
        return {}

    review_ids_list = list(review_ids)
    filter = {
        "$and": [
            {"sub_category": {"$eq": category}},
            {"primary_id": {"$in": review_ids_list}},
            {"review_count": { "$gt": 0 }}
        ]
    }
    if brand_label is not None: filter["$and"].append({"brand_label": {"$eq": brand_label}})

    candidates = product_collection.get(where=filter, include=['embeddings', 'metadatas'])
    cand_ids = candidates.get('ids')
    if cand_ids is None or len(cand_ids) == 0: return {}

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
        except: continue

    if not valid_idx: return {}

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
    if not label: return "상품을 먼저 선택해주세요", ""
    if label not in p_map: return f"오류: 선택한 제품 '{label}'의 데이터를 찾을 수 없습니다.", ""

    user_info = dict(user_info) if user_info else {}
    target_id = p_map[label]['id']
    cat = p_map[label]['cat']
    bl = p_map[label]['brand_lb']

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

    hyde_txt = llm.invoke(hyde_prompt.format(
        h=user_info.get('height',''), w=user_info.get('weight',''),
        g=user_info['gender'], p=user_info.get('p',''),
        p_info=label, cat=cat)).content
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
    height = int(user_info.get('height', 170))
    weight = int(user_info.get('weight', 65))

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

    if not final_docs_raw: return "유사 리뷰 데이터 부족", method_log

    best_doc, max_score, weight_docs = None, -1e9, []
    size_score = defaultdict(float)
    size_count = defaultdict(int)
    valid_doc_cnt = 0

    for doc in final_docs_raw:
        meta = getattr(doc, 'metadata', {}) or {}
        size = meta.get('size')
        if size:
            size_count[size] += 1
            valid_doc_cnt += 1

    if valid_doc_cnt == 0: return "best_doc is not found", method_log

    for doc in final_docs_raw:
        meta = getattr(doc, 'metadata', {}) or {}
        prod_id = meta.get('product_id')
        size = meta.get('size')
        if not prod_id or not size: continue

        dis = meta.get("score", 0.0)
        review_sim = max(0.0, 1.0 - dis)
        s_cnt = size_count[size]
        size_ratio_score = (1 + s_cnt / valid_doc_cnt)

        prod_id = str(prod_id)
        prod_sim = float(product_score_map.get(prod_id, 0.5))
        w_score = prod_sim * review_sim * size_ratio_score

        weight_docs.append({"doc": doc, "id": prod_id, "weight": w_score, "size": size})
        size_score[size] += w_score

        if w_score > max_score:
            max_score = w_score
            best_doc = {"product_id": prod_id, "size": size}

    if not best_doc: return "best_doc is not found", method_log

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
    evidence_txt = '\n'.join([
        f"-[{doc['size']}] {doc['doc'].page_content[:80]}...({doc['weight']:.2f})"
        for doc in weight_docs[:5]
    ])

    prompt = "패션 상품 구매를 고려하는 사용자{p}에게 {s} 사이즈만 추천한다. 다른 사이즈는 절대 언급하지 않는다.근거 {e}는 리뷰 데이터이며, 해당 리뷰 속에서 확인된 실제 착용감·핏·실루엣 관련 내용을 중심으로 간단하고 핵심적으로 요약하여 추천 근거를 생성한다. 비교 표현, 다른 사이즈 언급, 과도한 확장 설명은 사용하지 않는다. 근거:\n{e}\n페르소나: {p}"
    msg = llm.invoke(prompt.format(s=final_sz, e=evidence_txt, p=user_info.get('p',''))).content

    return final_sz, msg

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

# ==========================================
# [5] 기본 변수 및 UI 헬퍼 함수
# ==========================================
brand_description = {
    "미니멀리즘, 높은 퀄리티, 편안함, 타임리스, 웨어러블": "미니멀하고 타임리스한 디자인을 통해 높은 품질의 편안한 의류를 제공하며, 실용성과 웨어러블함을 중요시합니다.",
    "편안함, 스타일, 개성, 실용성, 자유로움": "이 클러스터의 브랜드들은 편안한 실루엣과 실용적인 디자인을 통해 개인의 개성을 표현하며, 일상에서 자유로운 스타일을 추구합니다.",
    "편안함, 일상, 유연함, 트렌디, 개인성": "일상 속에서 편안하게 착용할 수 있는 트렌디한 디자인을 추구하며, 개인의 개성과 다양한 라이프스타일을 반영하는 브랜드들이 모여 있습니다.",
    "편안함, 유니크함, 지속 가능성, 반항적 디자인, 일상성과 특별함": "이 클러스터의 브랜드들은 일상 속에서 편안함과 유니크함을 추구하며, 반항적이고 독창적인 디자인으로 새로운 흐름을 제안하는 동시에 지속 가능한 삶을 지향합니다.",
    "편안함, 개성, 자연스러움, 균형, 다채로움": "이 클러스터의 브랜드들은 편안함과 개성을 중시하며, 자연스러운 스타일과 균형 잡힌 라이프스타일을 통해 다양성을 존중하는 독창적인 패션을 제안합니다.",
    "아웃도어, 기능성, 일상, 스포츠, 혁신": "스포츠와 아웃도어 활동을 일상 속에서 즐길 수 있도록 기능성과 스타일을 겸비한 다양한 제품을 제공하는 브랜드들이 모인 클러스터입니다.",
    "여성, 클래식, 트렌디, 웨어러블, 심플, 감성": "이 클러스터의 브랜드들은 클래식함과 트렌디함을 조화롭게 담아내며, 편안한 착용감을 제공하는 웨어러블한 디자인과 감성적인 스타일을 강조합니다."
}

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
    if not text: return ""
    return f"""
    <div style="
        background-color: #f0f9ff; color: #333333; border: 1px solid #bae6fd;
        border-radius: 8px; padding: 12px; text-align: center;
        font-weight: 600; font-size: 0.95em; margin-bottom: 15px; margin-top: 5px;
    ">
        {text}
    </div>
    """

# ==========================================
# [6] Streamlit UI 구성
# ==========================================
st.set_page_config(page_title="패션 추천 시스템", page_icon="🛍️", layout="wide")

# 세션 상태 초기화
if 'page' not in st.session_state:
    st.session_state.page = 'input'  # 'input' 또는 'result'
if 'state_pmap' not in st.session_state:
    st.session_state.state_pmap = {}
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = ""
if 'rec_size' not in st.session_state:
    st.session_state.rec_size = ""
if 'rec_reason' not in st.session_state:
    st.session_state.rec_reason = ""
if 'user_info' not in st.session_state:
    st.session_state.user_info = {}

# ----------------- [페이지 1] 입력 화면 -----------------
if st.session_state.page == 'input':
    st.header("1. 스타일 조건을 입력해주세요 👕")
    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        in_g = st.radio("성별", ["male", "female"], index=0, horizontal=True)
    with col2:
        check = st.checkbox("다른 성별의 제품도 볼래요", value=False)

    col3, col4 = st.columns(2)
    with col3:
        in_h = st.number_input("키 (cm)", value=175.0, step=1.0)
    with col4:
        in_w = st.number_input("몸무게 (kg)", value=70.0, step=1.0)

    col5, col6 = st.columns(2)
    with col5:
        in_bg = st.selectbox("브랜드 그룹", brand_label_choice, help="선택 시 브랜드 설명이 아래에 나타납니다.")
        st.markdown(format_info_box(get_brand_desc(in_bg)), unsafe_allow_html=True)

    with col6:
        filtered_subcat = []
        if in_bg and not prod_data.empty:
            if 'subcategory' in prod_data.columns:
                filtered_subcat = sorted(prod_data[prod_data['brand_label'] == in_bg]['subcategory'].dropna().unique().tolist())
        in_cat = st.selectbox("카테고리", filtered_subcat if filtered_subcat else subcat_choice)

    cur_min_price, cur_max_price = min_price, max_price
    if in_bg and in_cat and not prod_data.empty:
        target_col = 'subcategory' if 'subcategory' in prod_data.columns else 'category_final'
        selected_prod_df = prod_data[(prod_data['brand_label'] == in_bg) & (prod_data[target_col] == in_cat)]
        if not selected_prod_df.empty:
            cur_min_price = int(selected_prod_df['sale_price'].min())
            cur_max_price = int(selected_prod_df['sale_price'].max())

    st.info(f"선택한 조건의 가격 범위: **{cur_min_price:,}원 ~ {cur_max_price:,}원**")

    col7, col8 = st.columns(2)
    with col7:
        min_num = st.number_input("최소 가격 입력", value=float(cur_min_price), step=1000.0)
    with col8:
        max_num = st.number_input("최대 가격 입력", value=float(cur_max_price), step=1000.0)

    if min_num < cur_min_price or max_num > cur_max_price or min_num > max_num:
        st.warning("⚠️ 가격 범위를 다시 확인해주세요.")

    in_sel = st.selectbox("스타일", style_key_options, help="원하는 스타일 무드를 선택해보세요.")
    st.markdown(format_info_box(get_style_desc(in_sel)), unsafe_allow_html=True)

    in_txt = st.text_input("텍스트 (패션 추구미를 직접 입력해보세요)", placeholder="예: 미니멀하면서 힙한 느낌")
    in_img = st.file_uploader("이미지를 넣으면 더 정밀한 추천을 받을 수 있어요 (선택 사항)", type=["png", "jpg", "jpeg"])

    st.divider()
    if st.button("🔍 검색 (사용자 데이터 분석 시작)", type="primary", use_container_width=True):
        if not in_bg or not in_cat:
            st.error("브랜드그룹과 카테고리를 반드시 선택하세요.")
        else:
            with st.spinner("🚀 사용자 데이터 분석을 시작합니다..."):
                img_path = None
                if in_img is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        tmp.write(in_img.getvalue())
                        img_path = tmp.name

                u_info = {'gender': in_g, 'height': in_h, 'weight': in_w}

                try:
                    gal_data, out_p_data, pmap_data, updated_u_info = search_recom_products(
                        in_bg, in_cat, in_sel, in_txt, img_path, u_info, min_num, max_num, check
                    )

                    try:
                        log_search_to_sheet(in_bg, in_cat, min_num, max_num, in_sel, in_txt, check, out_p_data)
                    except Exception as e:
                        st.error(f"구글 시트 저장 실패: {e}")

                    st.session_state.search_results = gal_data
                    st.session_state.state_pmap = pmap_data
                    st.session_state.user_info = updated_u_info
                    st.session_state.out_p_data = out_p_data
                    st.session_state.selected_product = ""
                    st.session_state.rec_size = ""
                    st.session_state.rec_reason = ""

                    st.session_state.page = 'result'
                    st.rerun()

                except Exception as e:
                    st.error(f"시스템 오류가 발생했습니다: {str(e)}")

# ----------------- [페이지 2] 추천 결과 화면 -----------------
elif st.session_state.page == 'result':
    st.header("2. 추천 결과 ✨")
    st.divider()

    st.text_area("나만의 패션 페르소나", value=st.session_state.get('out_p_data', ''), height=100, disabled=True)

    col_left, col_right = st.columns([1.5, 1])

    # 좌측: 갤러리 (추천 제품 목록)
    with col_left:
        st.subheader("검색 결과 (상품을 클릭하여 선택하세요)")
        gal_data = st.session_state.search_results

        if gal_data:
            cols = st.columns(3)
            for idx, (img_url, label) in enumerate(gal_data):
                with cols[idx % 3]:
                    if img_url:
                        st.image(img_url, use_container_width=True)
                    else:
                        st.info("이미지 없음")

                    if st.button("선택", key=f"select_{idx}", help=label):
                        st.session_state.selected_product = label
                        st.session_state.rec_size = ""
                        st.session_state.rec_reason = ""
                        st.rerun()
                    st.caption(label)
        else:
            st.info("조건에 맞는 상품이 없습니다.")

    # 우측: 사이즈 추천
    with col_right:
        st.subheader("사이즈 추천 📏")
        st.text_input("선택된 상품", value=st.session_state.selected_product, disabled=True)

        if st.button("사이즈 추천 받기", type="primary"):
            if not st.session_state.selected_product:
                st.warning("👈 왼쪽 갤러리에서 상품을 먼저 선택해주세요!")
            else:
                with st.spinner("체형 데이터를 기반으로 사이즈를 분석 중입니다..."):
                    try:
                        rec_size, reason = recom_size(
                            st.session_state.selected_product,
                            st.session_state.state_pmap,
                            st.session_state.user_info
                        )
                        st.session_state.rec_size = rec_size
                        st.session_state.rec_reason = reason

                        log_size_to_sheet(
                            st.session_state.selected_product,
                            st.session_state.user_info,
                            rec_size,
                            reason
                        )
                    except Exception as e:
                        st.error(f"사이즈 추천 중 오류 발생: {e}")

        st.text_input("추천 사이즈", value=st.session_state.rec_size, disabled=True)
        st.text_area("추천 이유 설명", value=st.session_state.rec_reason, height=150, disabled=True)

    st.divider()
    if st.button("⬅️ 검색 조건 다시 입력하기"):
        st.session_state.page = 'input'
        st.rerun()