import sqlite3
import streamlit as st
import google.generativeai as genai
import pandas as pd
import json

# ==========================================
# [환경 세팅] Streamlit Secrets를 이용한 안전한 API 키 연동
# ==========================================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
else:
    model = None


# ==========================================
# [AI 채점 로직] - 오직 "해석(번역)"만 평가함
# ==========================================
# 💡 선생님의 모범 정답(teacher_trans)을 파라미터로 추가로 받음
def check_translation_with_ai(original_sentence, student_trans, teacher_trans):
    if not model:
        return False

    # 💡 AI가 절대 헷갈리지 않도록 구체적인 예시와 기준을 명시
    prompt = f"""
    너는 영어 문장 번역을 채점하는 깐깐하지만 유연한 선생님이야. 
    다음 원문과 '모범 해석'을 기준으로, '학생 해석'이 올바른지 판단해.
    
    - 원문: {original_sentence}
    - 모범 해석(기준점): {teacher_trans}
    - 학생 해석: {student_trans}
    
    [채점 기준]
    1. 단어 단위: 모범 해석과 비교하여 핵심 의미가 같거나 유의어면 정답 인정. 아예 다른 맥락이거나 반대 의미면 오답.
    2. 조사 및 어미 단위 (한국어 문법):
       - [오답 처리]: 주어인데 목적격 조사(~을/를)를 쓰는 등 성분과 명백하게 맞지 않는 치명적인 조사 오류.
       - [정답 처리]: 사소한 맞춤법 실수, 띄어쓰기 오류, 또는 살짝 어색한 어미 변화(예: '~이다'를 '~음', '사랑한다'를 '사랑해'로 쓴 경우)는 의미가 통하면 무조건 정답 인정.
    
    위 기준을 종합하여 학생 해석이 타당하면 오직 "PASS", 치명적 오류가 있으면 오직 "FAIL"이라는 단어만 출력해.
    """
    try:
        response = model.generate_content(prompt)
        result_text = response.text.strip().upper()
        return "PASS" in result_text and "FAIL" not in result_text
    except Exception as e:
        return False


# ==========================================
# [데이터 가공 로직] - 태그 정리 및 마침표 추가
# ==========================================
def clean_tag_string(ta_str):
    if pd.isna(ta_str):
        return ""
    chunks = str(ta_str).split("/")
    new_chunks = []
    for c in chunks:
        c = c.strip()
        if c.lower() in ["(every)", "(often)"]:
            new_chunks.append("[Adv]")
        elif c.startswith("(") and c.endswith(")"):
            new_chunks.append("(Prep)")
        elif c == "[have]":
            new_chunks.append("M.V.")
        elif c == "[to]":
            new_chunks.append("(Prep)")
        elif c.lower() == "[adv]":
            new_chunks.append("[Adv]")
        else:
            c = c.replace("[", "").replace("]", "")
            new_chunks.append(c)
    return " / ".join(new_chunks)


def ensure_period(sentence):
    s = str(sentence).strip()
    if s and not s.endswith("."):
        s += "."
    return s


# ==========================================
# [DB 세팅] - 20개 문제 자동 생성 (외부 파일 불필요)
# ==========================================
def init_db():
    conn = sqlite3.connect("ta_database_v2.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_sentence TEXT,
            sentence TEXT NOT NULL,
            answer_ta TEXT NOT NULL,
            answer_translation TEXT,
            is_active INTEGER DEFAULT 0,
            set_name TEXT DEFAULT '기본 세트',
            order_num INTEGER DEFAULT 999
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS student_records (
            name TEXT PRIMARY KEY,
            a1_wrong INTEGER, a1_total INTEGER,
            a2_wrong INTEGER, a2_total INTEGER,
            a3_wrong INTEGER, a3_total INTEGER,
            status TEXT,
            answers_json TEXT,
            feedback_json TEXT
        )
    """)
    conn.commit()

    # 💡 [핵심 해결] DB가 비어있으면 20문제를 즉시 자동 삽입!
    c.execute("SELECT COUNT(*) FROM questions")
    if c.fetchone()[0] == 0:
        new_data = [
            # --- SET 1 ---
            (
                "Because of that, education becomes very important.",
                "(Because of / that), / education / becomes / very / important.",
                "prep / / S / Vl / adv / SC",
                "그것 때문에 교육은 매우 중요해진다.",
                "Set 1",
                1,
            ),
            (
                "Therefore, many Koreans think that producing good students is the best way to build a better future.",
                "Therefore, / many Koreans / think / [that / producing good students / is / the best way / to build / a better future].",
                "adv / S / Vt / / S / Vl / SC / adv(Vt) / O",
                "그러므로 많은 한국인들은 훌륭한 학생들을 배출하는 것이 더 나은 미래를 건설하는 최고의 방법이라고 생각한다.",
                "Set 1",
                2,
            ),
            (
                "That is why there is so much attention on good universities.",
                "That / is / why / there / is / so much attention / (on good universities).",
                "S / Vl / / adv / Vi / S / ",
                "그것이 좋은 대학에 그토록 많은 관심이 쏠리는 이유이다.",
                "Set 1",
                3,
            ),
            (
                "For a long time, Korean society has respected education and hard work.",
                "(For a long time), / Korean society / has respected / education and hard work.",
                "/ S / Vt / O",
                "오랫동안 한국 사회는 교육과 근면함을 존중해왔다.",
                "Set 1",
                4,
            ),
            (
                "In the past, people took big national exams to become government officers.",
                "(In the past), / people / took / big national exams / to become / government officers.",
                "/ S / Vt / O / adv(Vl) / SC",
                "과거에는 사람들이 공무원이 되기 위해 대규모 국가 시험을 치렀다.",
                "Set 1",
                5,
            ),
            # --- SET 2 ---
            (
                "It is true that going to a good university is not the only way to live a successful life.",
                "It / is / true / [that / going to a good university / is / not the only way / to live / a successful life].",
                "가S / Vl / SC / 진S / S / Vl / SC / adj(Vt) / O",
                "좋은 대학에 가는 것이 성공적인 삶을 사는 유일한 방법은 아니라는 것은 사실이다.",
                "Set 2",
                1,
            ),
            (
                "It gives you chances to meet excellent teachers, use better resources, and join strong networks that can help your future career.",
                "It / gives / you / chances / to meet / excellent teachers, / use / better resources, / and / join / strong networks / (that can / help / your future career).",
                "S / Vd / IO / DO / adj(Vt) / O / adj(Vt) / O / / adj(Vt) / O / / Vt / O",
                "그것은 당신에게 훌륭한 교사들을 만나고, 더 나은 자원을 사용하며, 미래의 경력에 도움이 될 수 있는 강력한 네트워크에 참여할 기회를 제공한다.",
                "Set 2",
                2,
            ),
            (
                "A degree from a respected school can help you get your first job faster, and it often gives you confidence that you have proved something important.",
                "A degree / (from a respected school) / can / help / you / get / your first job / faster, / and / it / often / gives / you / confidence / that / you / have proved / something / important.",
                "S / / m.v / VC / O / OC(Vt) / O / adv / / S / adv / Vd / IO / DO / / S / Vt / O / adj",
                "명문 학교의 학위는 첫 직장을 더 빨리 구하는 데 도움을 줄 수 있으며, 종종 중요한 무언가를 증명해냈다는 자신감을 준다.",
                "Set 2",
                3,
            ),
            (
                "So while success can come in many forms, studying hard for a good university is still one of the smartest and most practical ways to prepare for your future.",
                "So / while / success / can / come / (in many forms), / studying hard / (for a good university) / is / still / one of the smartest and most practical ways / to prepare / (for your future).",
                "adv / / S / m.v / Vi / / S / / Vl / adv / SC / adj(Vi) / ",
                "따라서 성공은 여러 형태로 올 수 있지만, 좋은 대학을 위해 열심히 공부하는 것은 여전히 미래를 준비하는 가장 현명하고 현실적인 방법 중 하나이다.",
                "Set 2",
                4,
            ),
            (
                "It may not guarantee happiness, but it gives you a powerful start in a competitive world - and that is something worth working for.",
                "It / may not / guarantee / happiness, / but / it / gives / you / a powerful start / (in a competitive world) - and / that / is / something / worth / (working for).",
                "S / m.v / Vt / O / / S / Vd / IO / DO / / S / Vl / SC / adj / 분구",
                "그것이 행복을 보장하지는 않을지 모르지만, 경쟁적인 세상에서 당신에게 강력한 출발을 제공하며 - 그것은 노력할 가치가 있는 것이다.",
                "Set 2",
                5,
            ),
            # --- SET 3 ---
            (
                "Runaway slaves couldn't trust just anyone along the Underground Railroad.",
                "Runaway slaves / couldn't / trust / just anyone / (along the Underground Railroad).",
                "S / m.v / Vt / O / ",
                "도망친 노예들은 지하 철도를 따라가면서 아무나 믿을 수는 없었다.",
                "Set 3",
                1,
            ),
            (
                "Fortunately, people were willing to risk their lives to help them.",
                "Fortunately, / people / were / willing / to risk / their lives / to help / them.",
                "adv / S / Vl / SC / adv / O / adv(Vt) / O",
                "다행히도, 사람들은 그들을 돕기 위해 기꺼이 목숨을 걸었다.",
                "Set 3",
                2,
            ),
            (
                "Coffin and his wife, Catherine, decided to make their home a station.",
                "Coffin and his wife, / (Catherine), / decided / to make / their home / a station.",
                "S / / Vt / O(VC) / O / OC",
                "코핀과 그의 아내 캐서린은 자신들의 집을 역으로 만들기로 결정했다.",
                "Set 3",
                3,
            ),
            (
                "More than 3000 slaves passed through their home heading north to Canada.",
                "More than 3000 slaves / passed / (through / their home) / heading / north / (to Canada).",
                "S / Vi / prep / p.o / 분구 / adv / ",
                "3,000명 이상의 노예들이 캐나다로 북상하며 그들의 집을 거쳐 지나갔다.",
                "Set 3",
                4,
            ),
            (
                "He hid runaways in his home in Rochester, New York, and helped 400 fugitives travel to Canada.",
                "He / hid / runaways / (in his home) (in Rochester, New York), and / helped / 400 fugitives / travel / (to Canada).",
                "S / Vt / O / / VC / O / OC /",
                "그는 뉴욕주 로체스터에 있는 자신의 집에 도망자들을 숨겨주었고, 400명의 도망자들이 캐나다로 이동하도록 도왔다.",
                "Set 3",
                5,
            ),
            # --- SET 4 ---
            (
                "By day he worked as a clerk for the Pennsylvania Anti-Slavery Society, but at night he secretly aided fugitives.",
                "(By day) / he / worked / (as a clerk) (for the Pennsylvania Anti-Slavery Society), but (at night) / he / secretly / aided / fugitives.",
                "/ S / Vi / / S / adv / Vt / O",
                "낮에는 그는 펜실베이니아 반노예제 협회의 서기로 일했지만, 밤에는 몰래 도망자들을 도왔다.",
                "Set 4",
                1,
            ),
            (
                "He raised money and helped hundreds of enslaved people escape to the North, but he also knew it was important to tell their stories.",
                "He / raised / money / and / helped / hundreds of enslaved people / escape / (to the North), but / he / also / knew / [it / was / important / to tell / their stories].",
                "S / Vt / O / / VC / O / OC / / S / adv / Vt / 가S / Vl / SC / 진S(Vt) / O",
                "그는 자금을 모아 수백 명의 노예가 북부로 탈출하도록 도왔지만, 그들의 이야기를 전하는 것이 중요하다는 것 또한 알고 있었다.",
                "Set 4",
                2,
            ),
            (
                "That 's why Still interviewed the runaways who came through his station, keeping detailed records of the individuals and families, and hiding his journals until after the Civil War.",
                "That / 's / why / Still / interviewed / the runaways / (who / came / (through / his station)), / keeping / detailed records of the individuals and families, / and / hiding / his journals / (until after the Civil War).",
                "S / Vl / / S / Vt / O / / Vi / prep / p.o / 분구(Vt) / O / / 분구(Vt) / O / ",
                "그것이 스틸이 자신의 역을 거쳐 온 도망자들을 인터뷰하면서 개인과 가족에 대한 자세한 기록을 남기고, 남북 전쟁이 끝날 때까지 자신의 일지를 숨긴 이유이다.",
                "Set 4",
                3,
            ),
            (
                "Then in 1872, he self-published his notes in his book, The Underground Railroad.",
                "Then / (in 1872), / he / self-published / his notes / (in his book), / The Underground Railroad.",
                "adv / / S / Vt / O / / N",
                "그 후 1872년에 그는 자신의 책 '지하 철도'에 자신의 노트를 자비로 출판했다.",
                "Set 4",
                4,
            ),
            (
                "It 's one of the clearest accounts of people involved with the Underground Railroad.",
                "It / 's / one of the clearest accounts of people / involved / (with the Underground Railroad). ",
                "S / Vl / SC / adj / ",
                "그것은 지하 철도와 관련된 사람들에 대한 가장 명확한 기록 중 하나이다.",
                "Set 4",
                5,
            ),
        ]
        c.executemany(
            "INSERT INTO questions (raw_sentence, sentence, answer_ta, answer_translation, set_name, order_num, is_active) VALUES (?, ?, ?, ?, ?, ?, 1)",
            new_data,
        )
        conn.commit()

    conn.close()


init_db()


def get_all_questions(only_active=False):
    conn = sqlite3.connect("ta_database_v2.db")
    if only_active:
        df = pd.read_sql_query(
            "SELECT * FROM questions WHERE is_active = 1 ORDER BY order_num ASC, id ASC",
            conn,
        )
    else:
        df = pd.read_sql_query(
            "SELECT * FROM questions ORDER BY set_name ASC, order_num ASC, id ASC", conn
        )
    conn.close()
    return df


def update_db_from_combined(df, set_active_states):
    if df.empty:
        return

    conn = sqlite3.connect("ta_database_v2.db")
    c = conn.cursor()

    try:
        c.execute("BEGIN TRANSACTION")
        c.execute("DELETE FROM questions")

        for _, row in df.iterrows():
            if "delete" in row and row["delete"] == True:
                continue
            if pd.isna(row["sentence"]) or str(row["sentence"]).strip() == "":
                continue
            if pd.isna(row["answer_ta"]) or str(row["answer_ta"]).strip() == "":
                continue

            set_name_val = str(row["set_name"]).strip()
            is_active_val = 1 if set_active_states.get(set_name_val, False) else 0
            ans_trans = (
                ""
                if pd.isna(row.get("answer_translation"))
                else str(row["answer_translation"])
            )

            sentence_val = ensure_period(row["sentence"])
            ta_val = clean_tag_string(row["answer_ta"])
            order_num_val = (
                int(row["order_num"]) if pd.notna(row.get("order_num")) else 999
            )

            # 💡 수정된 부분: None일 경우 괄호와 기호를 지운 클린 문장으로 자동 생성 후 DB에 영구 저장
            raw_val = (
                ""
                if pd.isna(row.get("raw_sentence"))
                else str(row["raw_sentence"]).strip()
            )

            if not raw_val or raw_val == "None":
                import re

                clean = re.sub(r"[()\[\]/]", " ", str(row["sentence"]))
                clean = re.sub(r"\s+", " ", clean).strip()
                raw_val = re.sub(r"\s+([.,?!])", r"\1", clean)

            # 💡 추가된 부분: INSERT 쿼리에 raw_sentence 완벽 반영
            c.execute(
                "INSERT INTO questions (id, raw_sentence, sentence, answer_ta, answer_translation, is_active, set_name, order_num) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row["id"],
                    raw_val,
                    sentence_val,
                    ta_val,
                    ans_trans,
                    is_active_val,
                    set_name_val,
                    order_num_val,
                ),
            )

        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        conn.close()


def save_student_record(
    name, attempt, wrong_count, total_count, status, answers_dict, feedback_dict
):
    conn = sqlite3.connect("ta_database_v2.db")
    c = conn.cursor()
    c.execute("SELECT * FROM student_records WHERE name=?", (name,))
    row = c.fetchone()

    ans_json = json.dumps(answers_dict)
    fb_json = json.dumps(feedback_dict)

    if not row:
        c.execute(
            "INSERT INTO student_records (name, a1_wrong, a1_total, status, answers_json, feedback_json) VALUES (?, ?, ?, ?, ?, ?)",
            (name, wrong_count, total_count, status, ans_json, fb_json),
        )
    else:
        if attempt == 1:
            c.execute(
                "UPDATE student_records SET a1_wrong=?, a1_total=?, status=?, answers_json=?, feedback_json=? WHERE name=?",
                (wrong_count, total_count, status, ans_json, fb_json, name),
            )
        elif attempt == 2:
            c.execute(
                "UPDATE student_records SET a2_wrong=?, a2_total=?, status=?, answers_json=?, feedback_json=? WHERE name=?",
                (wrong_count, total_count, status, ans_json, fb_json, name),
            )
        elif attempt == 3:
            c.execute(
                "UPDATE student_records SET a3_wrong=?, a3_total=?, status=?, answers_json=?, feedback_json=? WHERE name=?",
                (wrong_count, total_count, status, ans_json, fb_json, name),
            )
    conn.commit()
    conn.close()


def parse_grade(val):
    if pd.isna(val) or val == "미제출" or val == "":
        return None, None
    try:
        parts = str(val).split("/")
        return int(parts[0].strip()), int(parts[1].strip())
    except:
        return None, None


def update_records_from_dataframe(df):
    conn = sqlite3.connect("ta_database_v2.db")
    c = conn.cursor()
    c.execute("SELECT name, answers_json, feedback_json FROM student_records")
    existing_data = {row[0]: (row[1], row[2]) for row in c.fetchall()}
    c.execute("DELETE FROM student_records")

    for _, row in df.iterrows():
        if row.get("삭제 여부", False):
            continue
        name = str(row[" 이름"]).strip()
        if not name:
            continue

        a1_w, a1_t = parse_grade(row.get("1차 (오답/전체)"))
        a2_w, a2_t = parse_grade(row.get("2차 (오답/전체)"))
        a3_w, a3_t = parse_grade(row.get("3차 (오답/전체)"))

        # 💡 지능적인 상태 자동 업데이트 (수동 변경 시 상태 자동 연동)
        if a1_w == 0:
            status = "PASS_1"
        elif a2_w == 0:
            status = "PASS_2"
        elif a3_w == 0:
            status = "PASS_3"
        elif a3_w is not None and a3_w > 0:
            status = "FAIL"
        else:
            status = "PROGRESS"

        ans_json, fb_json = existing_data.get(name, ("{}", "{}"))
        c.execute(
            "INSERT INTO student_records (name, a1_wrong, a1_total, a2_wrong, a2_total, a3_wrong, a3_total, status, answers_json, feedback_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (name, a1_w, a1_t, a2_w, a2_t, a3_w, a3_t, status, ans_json, fb_json),
        )
    conn.commit()
    conn.close()


# ==========================================
# [화면 구성] Streamlit UI
# ==========================================
st.set_page_config(layout="wide", page_title="SETE TA 시스템 정식버전")
st.markdown(
    """
<style>
    h2 { font-size: 1.7rem !important; padding-bottom: 0.5rem; }
    h3 { font-size: 1.4rem !important; }
    .stTextInput input { padding: 12px; font-size: 16px; }
    .stDataFrame { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    .splash-container { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 70vh; }
    .main-title { font-size: 4.5rem; font-weight: 900; color: #1E3A8A; margin-bottom: 10px; }
    .sub-title { font-size: 1.8rem; color: #64748B; font-weight: 600; margin-bottom: 50px; }
    .lsb-footer { position: fixed; bottom: 15px; left: 20px; font-size: 0.85rem; color: #94A3B8; }
    .mode-title { text-align: center; color: #1E3A8A; margin-top: 100px; font-weight: 800; font-size: 2.5rem; }
    .chunk-box { text-align: center; font-size: 18px; font-weight: bold; border-bottom: 2px solid #1E3A8A; margin-bottom: 10px; padding-bottom: 5px; }
    .fb-pass { background-color:#d4edda; color:#155724; padding: 5px; border-radius: 5px; text-align: center; font-weight: bold; }
    .fb-fail { background-color:#f8d7da; color:#721c24; padding: 5px; border-radius:5px; text-align: center; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True,
)


def render_ta_guideline_popover():
    with st.popover("❔ 주요 TA 성분표"):
        st.markdown("""
        **[주어 / 동사류]**
        **S** / **가S** / **진S**
        **Vi** / **Vl**
        **Vt** / **Vd** / **VC** / **m.v**
        
        **[목적어 / 보어류]**
        **O** / **IO** / **DO** / **p.o**
        **SC** / **OC**
        
        **[수식어 및 기타]**
        **adj** / **adv** / **prep** / **분구**
        
        *(🚨 대소문자 무관. 복수 정답은 `분구(vt)` 형태로 자유롭게 기재)*
        """)


if "started" not in st.session_state:
    st.session_state.started = False
if "role" not in st.session_state:
    st.session_state.role = None

if not st.session_state.started:
    with st.container():
        st.markdown(
            """
        <div class="splash-container">
            <div class="main-title">SETE</div>
            <div class="sub-title">TA 자동 채점 시스템 (Ver 1.2)</div>
        </div>
        <div class="lsb-footer">made by LSB</div>
        """,
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("화면을 클릭하여 시작하기 🚀", use_container_width=True):
                st.session_state.started = True
                st.rerun()
    st.stop()

if st.session_state.role is None:
    with st.container():
        st.markdown(
            "<div class='mode-title'>Select Mode ⚙️</div>", unsafe_allow_html=True
        )
        st.write("")
        col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
        with col2:
            if st.button("👨‍🏫 선생님", use_container_width=True):
                st.session_state.role = "teacher_prompt"
                st.rerun()
        with col3:
            if st.button("👧 학생", use_container_width=True):
                st.session_state.role = "student"
                st.rerun()
    st.stop()

if st.session_state.role == "teacher_prompt":
    with st.container():
        st.markdown(
            "<h3 style='text-align:center;'>선생님 접근 권한 확인 🔑</h3>",
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            password = st.text_input(
                "암호를 입력해 주십시오.", type="password", key="login_password_input"
            )
            if password:
                if password == "1357":
                    st.session_state.role = "teacher"
                    st.rerun()
                else:
                    st.error("암호가 일치하지 않습니다.")
            if st.button("뒤로 가기 🔙", key="login_back_button"):
                st.session_state.role = None
                st.rerun()
    st.stop()

# ==========================================
# 선생님 모드
# ==========================================
if st.session_state.role == "teacher":
    if st.sidebar.button("모드 변경 🔄"):
        st.session_state.role = None
        st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📝 문제 출제", "📊 성적 현황", "📁 DB 관리 및 출제", "💡 시스템 Insight"]
    )

    with tab1:
        st.info(
            "👨‍🏫 **선생님 운영 가이드라인:** 이곳에서 수동으로 새로운 문제를 출제할 수 있습니다.\n\n"
            "🎯 **[원문장 추가]** 학생들에게 띄어쓰기 왜곡 없이 보여줄 '깔끔한 원문장'을 입력해 주십시오.\n\n"
            "✂️ **[구문 분석 문장]** `/`를 사용해 덩어리를 나눕니다. 괄호 `()`, `[]`의 짝을 반드시 맞춰주십시오. (빈칸 출제 시 `/ /` 입력)\n\n"
            "💡 **[복수 정답 허용]** `분구(vt)`처럼 괄호를 쓰면, 학생이 순서를 바꾸거나 띄어쓰기를 해도 모두 정답 처리됩니다! 💯"
        )
        col_t1, col_t2 = st.columns([4, 1])
        with col_t1:
            st.header("📝 문제 출제란")
        with col_t2:
            render_ta_guideline_popover()

        with st.form("teacher_input_form"):
            input_set_name = st.text_input(
                "세트 이름 (예: Set 5)", value="수동 출제 세트"
            )
            # 💡 예시 문장 Set 1의 1번 문장으로 변경
            input_raw = st.text_input(
                "원문장 (학생 화면 노출용) (예: Because of that, education becomes very important.)"
            )
            input_sentence = st.text_input(
                "구문 분석 문장 (예: (Because of / that), / education / becomes / very / important.)"
            )
            input_ta = st.text_input("TA 정답 (예: prep / / S / Vl / adv / SC)")
            input_trans = st.text_input(
                "해석 (예: 그것 때문에 교육은 매우 중요해진다.)"
            )

            if st.form_submit_button("DB에 문제 추가하기 ➕"):
                safe_raw = input_raw.strip()
                safe_sentence = input_sentence.strip()
                safe_ta = input_ta.strip()
                safe_set = input_set_name.strip()
                safe_trans = input_trans.strip()

                if (
                    not safe_raw
                    or not safe_sentence
                    or not safe_ta
                    or not safe_set
                    or not safe_trans
                ):
                    st.error(
                        "🚨 세트 이름, 원문장, 구문 분석 문장, TA 정답, 해석은 모두 입력해야 합니다."
                    )
                elif "/" not in safe_sentence or "/" not in safe_ta:
                    st.error(
                        "🚨 구문 분석 문장과 TA 정답에는 반드시 슬래시(/)가 포함되어야 합니다."
                    )
                else:
                    chunks_count = len(safe_sentence.split("/"))
                    ta_count = len(safe_ta.split("/"))

                    if chunks_count != ta_count:
                        st.error(
                            f"🚨 슬래시(/) 구문 개수가 다릅니다. (문장: {chunks_count}개, TA: {ta_count}개)"
                        )
                    else:
                        conn = sqlite3.connect("ta_database_v2.db")
                        c = conn.cursor()
                        # 💡 DB INSERT에 raw_sentence 추가
                        c.execute(
                            "INSERT INTO questions (raw_sentence, sentence, answer_ta, answer_translation, set_name, order_num) VALUES (?, ?, ?, ?, ?, 999)",
                            (safe_raw, safe_sentence, safe_ta, safe_trans, safe_set),
                        )
                        conn.commit()
                        conn.close()
                        st.success(f"🎉 '{safe_set}' 세트에 성공적으로 추가되었습니다.")
    with tab2:
        st.header("📊 성적 현황 및 상세 답안지")
        conn = sqlite3.connect("ta_database_v2.db")
        records_df = pd.read_sql_query("SELECT * FROM student_records", conn)
        conn.close()

        if not records_df.empty:
            disp_df = pd.DataFrame()
            disp_df["삭제 여부"] = False
            disp_df[" 이름"] = records_df["name"]

            # 💡 수정된 포맷 함수: 결측치(빈 값) 검사를 강화하여 사이트 다운 방지
            def fmt(w, t):
                if pd.isna(w) or pd.isna(t):
                    return "미제출"
                return f"{int(w)} / {int(t)}"

            disp_df["1차 (오답/전체)"] = records_df.apply(
                lambda x: fmt(x["a1_wrong"], x["a1_total"]), axis=1
            )
            disp_df["2차 (오답/전체)"] = records_df.apply(
                lambda x: (
                    "" if x["status"] == "PASS_1" else fmt(x["a2_wrong"], x["a2_total"])
                ),
                axis=1,
            )
            disp_df["3차 (오답/전체)"] = records_df.apply(
                lambda x: (
                    ""
                    if x["status"] in ["PASS_1", "PASS_2"]
                    else fmt(x["a3_wrong"], x["a3_total"])
                ),
                axis=1,
            )

            status_map = {
                "PASS_1": "🎉 1차 PASS",
                "PASS_2": "🎉 2차 PASS",
                "PASS_3": "🎉 3차 PASS",
                "FAIL": "🚨 선생님 호출",
            }
            disp_df[" 상태"] = records_df["status"].map(status_map).fillna("진행 중 🏃")

            edited_records = st.data_editor(
                disp_df,
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "삭제 여부": st.column_config.CheckboxColumn(
                        "🗑️ 삭제", default=False
                    )
                },
                use_container_width=True,
            )

            if st.button(
                "💾 성적 표 변경사항 저장 (삭제 및 상태 반영)", type="primary"
            ):
                update_records_from_dataframe(edited_records)
                st.success(
                    "업데이트 완료! 성적 변경에 따른 상태도 완벽하게 자동 연동되었습니다. ✨"
                )
                st.rerun()

            st.divider()
            st.subheader("학생 상세 답안지 조회 (최종 제출안)")
            selected_student = st.selectbox(
                "확인할 학생을 선택해 주세요.",
                ["선택 안함"] + records_df["name"].tolist(),
            )
            if selected_student != "선택 안함":
                student_data = records_df[records_df["name"] == selected_student].iloc[
                    0
                ]
                ans_dict = (
                    json.loads(student_data["answers_json"])
                    if student_data["answers_json"]
                    else {}
                )
                fb_dict = (
                    json.loads(student_data["feedback_json"])
                    if student_data["feedback_json"]
                    else {}
                )

                if not ans_dict:
                    st.write("아직 제출된 답안 내역이 없습니다. 📝")
                else:
                    active_questions_df = get_all_questions(only_active=True)
                    if not active_questions_df.empty:
                        with st.container(border=True):
                            for idx, row in active_questions_df.iterrows():
                                q_id = row["id"]
                                original_sentence = row["sentence"].replace("/", "")
                                st.markdown(f"**{idx + 1}. {original_sentence}**")

                                student_ta_combined = []
                                ans_ta_len = len(
                                    [t.strip() for t in row["answer_ta"].split("/")]
                                )

                                is_chunk_error = False
                                for i in range(ans_ta_len):
                                    if fb_dict.get(f"ta_{q_id}_{i}") == "CHUNK_ERROR":
                                        is_chunk_error = True
                                        break

                                if is_chunk_error:
                                    st.markdown(
                                        f"<div style='font-weight:bold; color:#721c24;'>학생 TA 제출안: ❌ 구문 나누기 오류 (정답과 의미 단위 개수가 다릅니다)</div>",
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    for i in range(ans_ta_len):
                                        input_key = f"ta_{q_id}_{i}"
                                        if input_key in ans_dict:
                                            tag = ans_dict[input_key]
                                            ta_pass = fb_dict.get(input_key, False)
                                            icon = "✅" if ta_pass == True else "❌"
                                            student_ta_combined.append(f"{icon} {tag}")
                                    ta_result_str = (
                                        " / ".join(student_ta_combined)
                                        if student_ta_combined
                                        else "(미제출)"
                                    )
                                    st.markdown(
                                        f"<div style='font-weight:bold;'>학생 TA 제출안: {ta_result_str}</div>",
                                        unsafe_allow_html=True,
                                    )

                                trans_key = f"trans_{q_id}"
                                trans_pass = fb_dict.get(trans_key, False)
                                st.markdown(
                                    f"<div style='color:{'#155724' if trans_pass else '#721c24'}; margin-top:5px;'>{'✅ 해석 통과' if trans_pass else '❌ 해석 재검토'}: {ans_dict.get(trans_key, '(미입력)')}</div><hr>",
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.warning(
                            "현재 활성화된 테스트 세트가 없어 문항을 불러올 수 없습니다. 🚨"
                        )
        else:
            st.write("응시 기록이 없습니다. 📝")

    with tab3:
        st.header("📁 전체 DB 관리 및 세트 출제")
        st.info(
            "세트(폴더)를 펼치고 '이 세트 출제하기' 체크박스를 켜면 즉시 시험지로 노출됩니다. '세트 이름'이나 '출제 순서'를 직접 수정하고 아래 '저장' 버튼을 누르면 즉시 이동 및 정렬이 반영됩니다! 📝"
        )
        db_df = get_all_questions(only_active=False)
        if not db_df.empty:
            db_df["order_num"] = (
                pd.to_numeric(db_df["order_num"], errors="coerce")
                .fillna(999)
                .astype(int)
            )

            all_sets = db_df["set_name"].unique()
            all_edited_dfs = []
            set_active_states = {}
            for s_name in all_sets:
                set_df = db_df[db_df["set_name"] == s_name].copy()
                is_set_active = bool(set_df["is_active"].sum() > 0)
                with st.expander(
                    f"📁 [{s_name}] (총 {len(set_df)}문제) - {'🟢 출제 중' if is_set_active else '⚪ 대기 중'}"
                ):
                    set_active_states[s_name] = st.checkbox(
                        f"이 세트 출제하기", value=is_set_active, key=f"chk_{s_name}"
                    )
                    set_df["delete"] = False

                    # 💡 표에 띄울 데이터 목록에 'raw_sentence' 추가 및 순서 재배치
                    edited_df = st.data_editor(
                        set_df[
                            [
                                "delete",
                                "order_num",
                                "set_name",
                                "raw_sentence",
                                "sentence",
                                "answer_ta",
                                "answer_translation",
                                "id",
                            ]
                        ],
                        column_config={
                            "delete": st.column_config.CheckboxColumn(
                                "🗑️ 삭제", default=False
                            ),
                            "order_num": st.column_config.NumberColumn(
                                "순서", min_value=1, max_value=999, step=1
                            ),
                            "set_name": st.column_config.TextColumn("세트 이름"),
                            "raw_sentence": st.column_config.TextColumn("원문장"),
                            "sentence": st.column_config.TextColumn("구문 분석 문장"),
                            "answer_ta": st.column_config.TextColumn("TA"),
                            "answer_translation": st.column_config.TextColumn("해석"),
                            "id": None,
                        },
                        hide_index=True,
                        use_container_width=True,
                        key=f"ed_{s_name}",
                    )
                    all_edited_dfs.append(edited_df)
            st.divider()

            # 💡 key="save_db_btn"을 추가하여 중복 에러 원천 차단
            if st.button(
                "💾 모든 세트 변경사항 DB에 적용", type="primary", key="save_db_btn"
            ):
                combined_df = pd.concat(all_edited_dfs, ignore_index=True)
                update_db_from_combined(combined_df, set_active_states)

                st.success(
                    "데이터베이스 성공적 업데이트! (순서 정렬, 세트 이동, 삭제 완벽 반영 완료) 🌟"
                )

                # 💡 5초 동안 화면을 대기시켜 성공 메시지를 유지한 뒤 새로고침
                import time

                time.sleep(5)
                st.rerun()

    with tab4:
        st.header("💡 시스템 Insight")
        st.info(
            "원활한 사이트 운영을 위해 시스템이 어떻게 작동하는지 핵심 기능을 정리해 드립니다. 🎯"
        )

        st.write("<br>", unsafe_allow_html=True)

        st.markdown("""
        ### 1. 📝 문제 출제 및 데이터 관리
        
        *   **원문장 자동 생성:** 학생들의 시험지에는 힌트가 될 수 있는 기호(`()`, `[]`, `/`)가 완전히 제거된 **깔끔한 원문장**이 노출됩니다. 선생님께서 문제 출제 시 구문 분석 문장 칸에만 기호를 넣어주시면 시스템이 똑똑하게 기호를 지워 **학생용 원문장**을 만들어 냅니다.
        
        *   **자유로운 복수 정답 인정:** `분구(vt)`처럼 괄호를 묶어 정답을 입력해 보세요. 학생들이 `vt(분구)`로 순서를 바꾸어 적거나 띄어쓰기를 다르게 하더라도 시스템이 의미를 파악해 **모두 정답으로 인정**합니다.
        
        *   **원클릭 시험지 배포:** [📁 DB 관리 및 출제] 탭에서 특정 세트를 펼치고 **이 세트 출제하기** 체크박스를 켜면 즉시 학생들의 화면에 해당 문제가 나타납니다.
        """)

        st.write("<br>", unsafe_allow_html=True)
        st.divider()
        st.write("<br>", unsafe_allow_html=True)

        st.markdown("""
        ### 2. 👧 학생 응시 및 똑똑한 채점 원리
        
        *   **스마트 구문 보정 (Snap 로직):** 학생들이 덩어리를 나누는 슬래시(`/`)를 올바른 위치에 넣었다면, 괄호 `()`, `[]`의 자리를 띄어쓰기 등으로 조금 헷갈리게 입력하더라도 선생님이 출제하신 **깔끔한 형태의 덩어리로 자동 보정**되어 화면에 통일감 있게 나타납니다.
        
        *   **3단계 방어막 검증:** 학생들이 문장의 단어를 훼손하거나(임의 삭제/변경), 괄호의 짝을 잘못 맞추어 입력하면 시스템이 **즉시 경고창을 띄워** 잘못된 오답 제출을 사전에 방지합니다.
        
        *   **AI 기반 해석 채점:** 학생이 작성한 해석은 AI 선생님이 **문맥과 핵심 의미를 파악하여 유연하게 채점**합니다. 사소한 맞춤법이나 어미의 차이는 정답으로 인정하지만, 주어/목적어를 헷갈리는 등 **치명적인 오역은 깐깐하게 오답 처리**합니다.
        """)

        st.write("<br>", unsafe_allow_html=True)
        st.divider()
        st.write("<br>", unsafe_allow_html=True)

        st.markdown("""
        ### 3. 📊 성적 현황 및 피드백 연동
        
        *   **실시간 상태 업데이트:** 학생들의 응시 횟수(1차~3차)와 채점 결과에 따라 상태가 `PASS_1`, `FAIL` 등으로 표에 **자동으로 연동**되어 표기됩니다.
        
        *   **수동 성적 조정:** 선생님께서 학생의 성적을 표에서 직접 더블클릭하여 수정하고 저장하시면, 시스템이 바뀐 점수에 맞춰 학생의 합격/불합격 상태도 똑똑하게 **자동으로 업데이트**합니다.
        """)

# ==========================================
# 학생 모드
# ==========================================
elif st.session_state.role == "student":
    if st.sidebar.button("모드 변경 🔄"):
        st.session_state.role = None
        st.session_state.pop("student_name", None)
        st.rerun()

    if "student_name" not in st.session_state:
        st.session_state.student_name = ""
    if "attempt" not in st.session_state:
        st.session_state.attempt = 1
    if "test_status" not in st.session_state:
        st.session_state.test_status = "PROGRESS"
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}
    if "user_inputs" not in st.session_state:
        st.session_state.user_inputs = {}

    if not st.session_state.student_name:
        st.subheader("이름을 입력해 주세요. 🏷️")

        # 💡 form으로 묶어서 Enter 키 입력 시 자동으로 제출되도록 수정
        with st.form("student_name_form"):
            name_input = st.text_input("이름", label_visibility="collapsed")
            submitted = st.form_submit_button("다음 ➡️")

            if submitted:
                if name_input.strip():
                    st.session_state.student_name = name_input.strip()
                    st.rerun()
                else:
                    st.error("이름을 반드시 입력해 주셔야 합니다. 😅")
    else:
        st.info(
            "🎓 **학생 응시 가이드라인:** 원문장을 읽고, 아래 입력창에서 슬래시(`/`)로 구문을 나누고 성분을 채우십시오.\n\n"
            "🧩 **[기호 사용]** 괄호 `()`, `[]`가 있다면 짝을 정확히 맞춰주십시오.\n\n"
            "👻 **[투명 빈칸]** 채울 성분이 없는 칸은 아무것도 적지 말고 그대로 제출하십시오.\n\n"
            "✌️ **[복수 정답]** 한 칸에 여러 성분이 겹친다면 `분구(vt)`처럼 괄호를 묶어 자유롭게 적어주십시오! (순서 상관없음)"
        )

        col_s1, col_s2 = st.columns([5, 1])
        with col_s1:
            st.markdown(
                f"### 👧 {st.session_state.student_name} 학생의 TA 테스트 (현재 {st.session_state.attempt}차)"
            )
        with col_s2:
            render_ta_guideline_popover()

        if st.session_state.test_status.startswith("PASS"):
            st.success(
                f"🎉 {st.session_state.test_status.split('_')[1]}차 시도 최종 통과! 고생했어요! 🎉"
            )
            if st.button("처음으로 돌아가기 🔙"):
                st.session_state.clear()
                st.rerun()
            st.stop()
        if st.session_state.test_status == "FAIL":
            st.error(
                "🚨 **선생님 호출:** 오답 횟수를 초과했습니다. 선생님께 와서 피드백을 받아주세요! 😊"
            )
            if st.button("처음으로 돌아가기 🔙"):
                st.session_state.clear()
                st.rerun()
            st.stop()

        active_questions_df = get_all_questions(only_active=True)
        if active_questions_df.empty:
            st.write("현재 출제된 테스트 세트가 없습니다. 잠시만 기다려 주세요. ☕")
            st.stop()

        local_inputs = {}
        total_questions = len(active_questions_df)

        for idx, row in active_questions_df.iterrows():
            q_id = row["id"]

            # 💡 수정된 부분: 선생님의 괄호가 포함된 sentence 대신 완벽히 클린한 raw_sentence 사용
            raw_sentence = str(row.get("raw_sentence", "")).strip()

            # DB 문제로 None이 들어올 경우를 대비한 자동 클린 방어 코드
            if not raw_sentence or raw_sentence == "None":
                import re

                raw_sentence = re.sub(r"[()\[\]/]", " ", str(row["sentence"]))
                raw_sentence = re.sub(r"\s+", " ", raw_sentence).strip()
                raw_sentence = re.sub(r"\s+([.,?!])", r"\1", raw_sentence)

            q_sentence_clean = raw_sentence

            st.markdown(f"#### {idx + 1}. {q_sentence_clean}")

            chunk_input_key = f"chunk_input_{q_id}"
            if chunk_input_key not in st.session_state:
                st.session_state[chunk_input_key] = q_sentence_clean

            student_chunked = st.text_input(
                "👉 위 문장 사이사이에 슬래시(/)를 넣어 구문을 나누고 엔터를 치십시오.",
                key=chunk_input_key,
            )

            # 💡 에러 방지를 위해 덩어리 변수를 미리 빈 리스트로 초기화
            chunks = []

            # 💡 [핵심 수정] 초기 상태(학생이 안 건드린 상태)와 채점 상태를 완벽 분리!
            if student_chunked == q_sentence_clean:
                # 1. 초기 상태: 원문을 그대로 1개 덩어리로 표시 (에러 팝업 없음, 해석 칸 정상 노출)
                chunks = [q_sentence_clean]
            elif student_chunked:
                import re

                # 2. 훼손 검사용 (알파벳, 숫자만 추출하여 띄어쓰기/기호 무시)
                orig_base = re.sub(r"[^a-zA-Z0-9]", "", q_sentence_clean).lower()
                stud_base = re.sub(r"[^a-zA-Z0-9]", "", student_chunked).lower()

                if orig_base != stud_base:
                    st.error(
                        "🚨 문장의 원본 단어가 변경되었습니다! 알파벳이나 숫자를 임의로 지우거나 수정하지 말아주세요. 📝"
                    )
                else:
                    # 3. 괄호 () [] 검사용 (알파벳, 숫자, 괄호 기호만 남겨서 짝패 비교)
                    teacher_brackets_text = re.sub(
                        r"[^a-zA-Z0-9\(\)\[\]]", "", str(row["sentence"])
                    ).lower()
                    student_brackets_text = re.sub(
                        r"[^a-zA-Z0-9\(\)\[\]]", "", student_chunked
                    ).lower()

                    if teacher_brackets_text != student_brackets_text:
                        st.error(
                            "🚨 괄호 `()`, `[]`의 위치나 개수가 정확하지 않습니다! 묶어야 할 구문 성분을 다시 한번 꼼꼼히 확인해 주세요. 🧐"
                        )
                    else:
                        # 4. 정상 통과 (에러가 없을 때만 슬래시 기준으로 덩어리 나뉨)
                        def get_core_chunks(text):
                            t = text.lower()
                            for c in " ',.:;-()[]?!":
                                t = t.replace(c, "")
                            return [x.strip() for x in t.split("/")]

                        if get_core_chunks(student_chunked) == get_core_chunks(
                            row["sentence"]
                        ):
                            student_chunked_clean = row["sentence"]
                        else:
                            student_chunked_clean = student_chunked

                        chunks = student_chunked_clean.split("/")

            # 💡 chunks 리스트가 성공적으로 생성되었을 때만 화면 렌더링
            if len(chunks) > 0:
                cols = st.columns(len(chunks))
                for i, chunk in enumerate(chunks):
                    with cols[i]:
                        st.markdown(
                            f"<div class='chunk-box'>{chunk.strip()}</div>",
                            unsafe_allow_html=True,
                        )
                        input_key = f"ta_{q_id}_{i}"

                        if input_key in st.session_state.feedback:
                            feedback_val = st.session_state.feedback[input_key]
                            if feedback_val == True:
                                st.markdown(
                                    "<div class='fb-pass'>✅ 정답</div>",
                                    unsafe_allow_html=True,
                                )
                            elif feedback_val == "CHUNK_ERROR":
                                st.markdown(
                                    "<div class='fb-fail'>✂️ 구문 분석 오류</div>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    "<div class='fb-fail'>❌ 오답</div>",
                                    unsafe_allow_html=True,
                                )

                        default_val = st.session_state.user_inputs.get(input_key, "")
                        val = st.text_input(
                            "태그 입력",
                            value=default_val,
                            key=f"input_ta_{q_id}_{i}",
                            label_visibility="collapsed",
                        )
                        local_inputs[input_key] = val

                trans_key = f"trans_{q_id}"
                if trans_key in st.session_state.feedback:
                    if st.session_state.feedback[trans_key] == True:
                        st.markdown(
                            "<div class='fb-pass'>✅ 해석 완벽!</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            "<div class='fb-fail'>❌ 해석 오답</div>",
                            unsafe_allow_html=True,
                        )

                default_trans = st.session_state.user_inputs.get(trans_key, "")
                trans_val = st.text_input(
                    "✍️ 문장 전체 해석을 적어주십시오.",
                    value=default_trans,
                    key=f"input_trans_{q_id}",
                )
                local_inputs[trans_key] = trans_val

            st.divider()

        if st.button(
            f"🚀 {st.session_state.attempt}차 제출 및 채점하기", type="primary"
        ):
            is_altered = False

            for _, row in active_questions_df.iterrows():
                q_id = row["id"]
                raw_sentence = str(row.get("raw_sentence", "")).strip()

                # DB 문제로 None이 들어올 경우를 대비한 자동 클린 방어 코드 (제출 시에도 적용)
                if not raw_sentence or raw_sentence == "None":
                    import re

                    raw_sentence = re.sub(r"[()\[\]/]", " ", str(row["sentence"]))
                    raw_sentence = re.sub(r"\s+", " ", raw_sentence).strip()
                    raw_sentence = re.sub(r"\s+([.,?!])", r"\1", raw_sentence)

                student_chunked_now = st.session_state.get(f"chunk_input_{q_id}", "")

                if not student_chunked_now:
                    continue  # 미입력 상태면 방어막 패스 (어차피 빈칸 오답 처리됨)

                import re

                orig_base = re.sub(r"[^a-zA-Z0-9]", "", raw_sentence).lower()
                stud_base = re.sub(r"[^a-zA-Z0-9]", "", student_chunked_now).lower()

                if orig_base != stud_base:
                    is_altered = True
                    break

            if is_altered:
                st.error(
                    "🚨 문장의 원본 단어가 훼손된 곳이 있습니다! 다시 한번 꼼꼼히 확인한 후 제출해 주세요. 📝"
                )
                st.stop()

            wrong_count = 0
            feedback_dict = {}
            answers_dict = {}

            st.session_state.user_inputs.update(local_inputs)

            with st.spinner(
                "AI 선생님이 문맥과 구조를 꼼꼼히 채점 중입니다... 잠시만 기다려 주십시오. 🤖"
            ):
                for _, row in active_questions_df.iterrows():
                    q_id = row["id"]
                    q_sentence = row["sentence"].replace("/", "")
                    ans_ta = row["answer_ta"]
                    ta_answers = [t.strip().lower() for t in ans_ta.split("/")]

                    student_chunked_now = st.session_state.get(
                        f"chunk_input_{q_id}", ""
                    )

                    # 💡 제출 시에도 동일한 스냅 로직 적용하여 정확한 채점 덩어리 산정
                    def get_core_chunks(text):
                        t = text.lower()
                        for c in " ',.:;-()[]?!":
                            t = t.replace(c, "")
                        return t.split("/")

                    if get_core_chunks(student_chunked_now) == get_core_chunks(
                        row["sentence"]
                    ):
                        student_chunked_clean_now = row["sentence"]
                    else:
                        student_chunked_clean_now = student_chunked_now

                    student_chunks_count = len(student_chunked_clean_now.split("/"))

                    is_question_wrong = False

                    if student_chunks_count != len(ta_answers):
                        is_question_wrong = True
                        for i in range(student_chunks_count):
                            input_key = f"ta_{q_id}_{i}"
                            student_ans = (
                                local_inputs.get(input_key, "").strip().lower()
                            )
                            answers_dict[input_key] = student_ans
                            feedback_dict[input_key] = "CHUNK_ERROR"
                    else:
                        for i, correct_ta in enumerate(ta_answers):
                            input_key = f"ta_{q_id}_{i}"
                            student_ans = (
                                local_inputs.get(input_key, "").strip().lower()
                            )
                            answers_dict[input_key] = student_ans

                            # 💡 복수 정답(괄호) 집합(Set) 분리 및 공백 완벽 제거 로직
                            correct_ta_clean = correct_ta.replace(" ", "")
                            student_ans_clean = student_ans.replace(" ", "")

                            teacher_opts = set(
                                p
                                for p in correct_ta_clean.replace(")", "(").split("(")
                                if p
                            )
                            student_opts = set(
                                p
                                for p in student_ans_clean.replace(")", "(").split("(")
                                if p
                            )

                            if not teacher_opts:  # 선생님이 정답 칸을 비워둔 경우
                                if (
                                    student_opts
                                ):  # 학생이 빈칸에 무언가 적었다면 오답 처리
                                    feedback_dict[input_key] = False
                                    is_question_wrong = True
                                else:
                                    feedback_dict[input_key] = True
                            else:  # 선생님 정답이 존재하는 경우
                                if not student_opts:  # 학생이 칸을 비워두면 오답 처리
                                    feedback_dict[input_key] = False
                                    is_question_wrong = True
                                else:
                                    # 💡 학생의 모든 입력값이 선생님 정답 바구니(Set) 안에 포함되는지 완벽 검증
                                    if student_opts.issubset(teacher_opts):
                                        feedback_dict[input_key] = True
                                    else:
                                        feedback_dict[input_key] = False
                                        is_question_wrong = True

                    # 💡 [핵심 복구 포인트 1] 해석 채점은 if/else 블록에서 빠져나와 무조건 실행됨!
                    trans_key = f"trans_{q_id}"
                    student_trans = local_inputs.get(trans_key, "").strip()
                    answers_dict[trans_key] = student_trans

                    if not student_trans:
                        feedback_dict[trans_key] = False
                        is_question_wrong = True
                    else:
                        # 💡 기존 로직에 row['answer_translation'] (선생님 모범 정답)을 추가로 넘겨줌!
                        is_trans_correct = check_translation_with_ai(
                            q_sentence, student_trans, str(row["answer_translation"])
                        )
                        feedback_dict[trans_key] = is_trans_correct
                        if not is_trans_correct:
                            is_question_wrong = True

                    # 💡 [핵심 복구 포인트 2] 오답 카운트도 무조건 실행됨!
                    if is_question_wrong:
                        wrong_count += 1

                # 💡 [핵심 복구 포인트 3] for문(문제 반복)이 끝난 후 아래 30줄 정상 실행!
                st.session_state.feedback = feedback_dict

                current_attempt = st.session_state.attempt
                if wrong_count == 0:
                    status = f"PASS_{current_attempt}"
                elif current_attempt >= 3:
                    status = "FAIL"
                else:
                    status = f"PROGRESS_{current_attempt}"

                save_student_record(
                    st.session_state.student_name,
                    current_attempt,
                    wrong_count,
                    total_questions,
                    status,
                    answers_dict,
                    feedback_dict,
                )

                st.session_state.test_status = status
                if status.startswith("PROGRESS"):
                    st.session_state.attempt += 1

                st.rerun()
