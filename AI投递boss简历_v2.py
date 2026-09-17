# -*- coding: utf-8 -*-
"""
AI 投递助手 v2 —— 双赛道版（AI应用 / 自动化测试）
=================================================

相比 v1 的核心改动：
  1. 【修 bug】v1 读 boss_jobs_*.json，但那里面没有 jd 字段，导致占 35 分权重的
     语义匹配从来没真正跑过，一直是兜底分。v2 会把 boss_details_*.json 的 JD
     按 job_id 合并进来，语义匹配正式开始工作。
  2. 【拆画像】AI应用 / 自动化测试 两套独立 PROFILE，技术栈、薪资、主打项目全分开。
  3. 【招呼语路由】不再"默认讲爬虫"——按岗位类别选最贴的那个项目来讲。
  4. 【双输出】AI 赛道和 QA 赛道各自一份 CSV + 一份 HTML 面板，互不干扰。
  5. 【安全】API key 改读环境变量；面板加"今日已投"计数提示。
  6. 【人工投递】本脚本不自动投递、不模拟点击、不自动发送任何消息。
     它只做：本地匹配排序 -> 生成招呼语 -> 给你一个手动点击的面板。

用法：
    set DEEPSEEK_API_KEY=sk-xxxx        （不设也行，会自动降级为模板招呼语）
    python AI投递boss简历_v2.py
"""

import os
import re
import ast
import glob
import json
import html
import sys
import datetime

# Windows 控制台默认 GBK，输出 emoji 会直接崩，这里强制 UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.expanduser(r"~\.boss-zhipin-scraper\job-result")

# 只合并最近 N 天内抓取过的数据（避免把上个月的过期岗位翻出来）
LOOKBACK_DAYS = 14

# ============================================================
# 第一步：两套个人简历画像
# ============================================================

PROFILE_AI = {
    'track': 'ai',
    'label': 'AI应用',
    'tech_skills': ['Python', 'LangChain', 'Chroma', 'RAG', 'Embedding', 'Prompt',
                    'Prompt Engineering', '大模型', 'LLM', 'AI Agent', 'Agent', '智能体',
                    'API', 'Pandas', 'NumPy', 'MySQL', 'SQL', '向量数据库', '语义检索',
                    '知识库', 'Function Calling', 'MCP'],
    'years_exp': 0.0,
    'education': '本科',
    'expected_salary': 15,
    # 命中这些关键词的岗位加分（越贴你的项目越靠前）
    'bonus_keywords': ['RAG', 'LangChain', 'Chroma', '向量', '知识库', 'Agent', '智能体',
                       'Prompt', '大模型', 'LLM', '语义检索', 'Embedding', 'AI应用'],
    # 命中这些的扣分（明显不是你要的方向）
    'penalty_keywords': ['算法工程师', '深度学习', '模型训练', '微调', 'PyTorch', 'TensorFlow',
                         'NLP算法', 'CV算法', '推荐算法', '硕士'],
    'personal_summary': """
我是马丁，2026届电子信息工程本科毕业生，可立即到岗，期望城市北京。

核心项目：
1. RAG论文问答系统（8月，已开源）：基于 LangChain + Chroma + BGE Embedding 独立开发，
   支持 PDF 上传、文本切分、向量化存储、语义检索与答案引用溯源，完整走通 RAG 全链路；
2. 智能求职助手（9月）：Chrome DevTools Protocol 采集岗位数据，TF-IDF + 余弦相似度做匹配排序，
   调用大模型 API 生成差异化话术，并用 HTML 面板承载，形成完整 AI 应用闭环；
3. AI图像风格统一生成（6月）：设计结构化 Prompt 模板（光源/色调/景别/场景多维度控制），
   将风格统一度从 60% 提升至 90% 以上，沉淀 30 项一致性检查 SOP；
4. 金融终端自动化导出系统（6月）：图像识别 + 模拟操作，实现无人值守定时导出与邮件投递。

技术栈：Python、LangChain、Chroma、BGE Embedding、Prompt Engineering、
大模型 API 调用（DeepSeek/ChatGLM）、Pandas、MySQL、自动化脚本。
""",
    # 招呼语默认讲哪个项目（按 JD 关键词再切换）
    'main_project': 'RAG论文问答系统',
    'main_project_desc': '基于 LangChain + Chroma 搭建的 RAG 系统，走通了文档加载、文本切分、'
                         '向量化存储、语义检索到答案溯源的全链路，能独立把大模型能力接进业务系统',
    'fallback_project': 'AI图像风格统一生成',
    'greeting_style': 'professional',
}

PROFILE_QA = {
    'track': 'qa',
    'label': '自动化测试',
    'tech_skills': ['Python', 'pytest', 'unittest', 'Selenium', 'Playwright', 'Appium',
                    'requests', '接口测试', '自动化测试', '测试开发', '测试用例', 'Postman',
                    'JMeter', 'MySQL', 'SQL', 'Linux', 'Git', 'Jenkins', 'CI', '持续集成',
                    'UI自动化', '性能测试', '抓包', 'Charles', 'Fiddler'],
    'years_exp': 0.0,
    'education': '本科',
    'expected_salary': 12,
    'bonus_keywords': ['自动化测试', '测试开发', '测试工程师', 'pytest', 'Selenium', 'Playwright',
                       'Appium', '接口测试', 'UI自动化', '质量保障', 'QA', 'SDET',
                       '测试用例', '性能测试', '持续集成', 'Jenkins'],
    'penalty_keywords': ['硕士', '5年以上', '10年以上', '硬件测试', '芯片测试', 'EMC',
                         '结构测试', '可靠性测试', '化学', '材料'],
    # 标题命中这些词直接剔除（不做扣分，避免高分岗又冒回来）
    # 方向：硬件/车载/嵌入式 类的测试岗，跟电子信息软件方向不匹配
    'exclude_title_pattern': (
        r'车载|整车|电机|电池|BMS|嵌入式|单片机|硬件测试|射频|天线|'
        r'算法测试|外场测试|EMC|结构测试|可靠性测试|'
        r'智驾|行车测试|辅助驾驶|自动驾驶'
    ),
    'personal_summary': """
我是马丁，2026届电子信息工程本科毕业生，可立即到岗，期望城市北京。

核心项目：
1. 金融终端自动化导出系统（6月）：独立设计并实现无人值守的桌面自动化程序。针对交易客户端
   界面复杂、每日导出覆盖旧文件、人工重复操作繁琐的痛点，通过图像识别定位界面元素、
   模拟键鼠操作，"图像识别 + 模拟操作"实现定时任务，并主动处理了弹窗等异常分支；
   在 Excel 中自动加时间戳归档解决覆盖问题，再开发合并模块按日期汇总当天数据，
   最后邮件自动发送；打包成 exe，非技术同事双击即用，上线后每天节省 30 分钟人工操作；
2. 智能求职助手（9月）：基于 Chrome DevTools Protocol 控制浏览器，实现稳定的数据采集
   与字段结构化，并用 TF-IDF + 余弦相似度做匹配排序，HTML 面板承载结果；
3. RAG论文问答系统（8月，已开源）：LangChain + Chroma，完整走通检索增强生成链路。

技术栈：Python、pytest、Selenium、requests、接口测试、SQL、
自动化脚本与定时任务、Linux/Git 基础。
""",
    'main_project': '金融终端自动化导出系统',
    'main_project_desc': '用「图像识别定位元素 + 模拟键鼠」实现无人值守的定时导出，'
                         '主动处理了弹窗异常等边缘情况，并打包成 exe 让非技术同事直接用，'
                         '上线后每天省 30 分钟人工操作、数据零遗漏',
    'fallback_project': '智能求职助手（CDP采集 + 匹配排序）',
    'greeting_style': 'professional',
}

ALL_PROFILES = [PROFILE_AI, PROFILE_QA]

# ============================================================
# 第二步：DeepSeek API 配置
#
#   key 存在同目录的 .env 文件里（该文件已加入 .gitignore，不会提交 Git）：
#       DEEPSEEK_API_KEY=sk-xxxx
#       DEEPSEEK_MODEL=deepseek-chat
#
#   换 key 只需要改 .env 那一行，本文件不用动。
#   也兼容直接设系统环境变量（.env 里的值会覆盖系统环境变量）。
# ============================================================
ENV_PATH = os.path.join(SCRIPT_DIR, '.env')

try:
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH, override=True)
except ImportError:
    # 没装 python-dotenv 时的手工兜底，保证不装包也能跑
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, 'r', encoding='utf-8') as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith('#') or '=' not in _line:
                    continue
                _k, _, _v = _line.partition('=')
                os.environ[_k.strip()] = _v.strip().strip('"').strip("'")

DEEPSEEK_CONFIG = {
    'api_key': os.environ.get('DEEPSEEK_API_KEY', '').strip(),
    'model': os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat'),
    'base_url': 'https://api.deepseek.com/v1/chat/completions',
}

# ============================================================
# 岗位分类：判断一个岗位属于哪个赛道
# ============================================================
QA_TITLE_PATTERN = re.compile(
    r'测试开发|自动化测试|软件测试|测试工程师|测试专家|QA|质量保障|质量工程|SDET|'
    r'测试脚本|测试平台|测试架构',
    re.IGNORECASE,
)
AI_TITLE_PATTERN = re.compile(
    r'AI|人工智能|大模型|LLM|算法应用|Agent|智能体|RAG|AIGC|Prompt|机器学习|'
    r'数据挖掘|自然语言|NLP',
    re.IGNORECASE,
)


def classify_track(row):
    """返回 'qa' / 'ai' / None（不关心的岗位）"""
    title = str(row.get('title', '') or '')
    industry = str(row.get('company_industry', '') or '')
    jd = str(row.get('jd', '') or '')

    # 测试岗优先级最高：标题里出现测试关键词，直接归 QA
    if QA_TITLE_PATTERN.search(title):
        return 'qa'
    # JD 里大量出现测试关键词、且标题不是明显 AI 岗时，也归 QA
    if not AI_TITLE_PATTERN.search(title):
        qa_hits = sum(1 for k in ['自动化测试', '测试用例', 'pytest', 'Selenium', '接口测试',
                                  '测试开发', '缺陷', '质量保障', 'Playwright', 'Appium']
                      if k.lower() in jd.lower())
        if qa_hits >= 4:
            return 'qa'
    if AI_TITLE_PATTERN.search(title) or AI_TITLE_PATTERN.search(industry):
        return 'ai'
    return None


# ============================================================
# 第三步：加载并合并数据（岗位列表 + JD 详情）
# ============================================================
def _load_recent(pattern):
    """读取最近 LOOKBACK_DAYS 天内的所有匹配文件，返回 (记录列表, 文件列表)"""
    files = glob.glob(os.path.join(RESULT_DIR, pattern))
    files = [f for f in files if 'merged' not in os.path.basename(f)]
    cutoff = datetime.datetime.now() - datetime.timedelta(days=LOOKBACK_DAYS)
    recent = []
    for f in files:
        try:
            mt = datetime.datetime.fromtimestamp(os.path.getmtime(f))
        except OSError:
            continue
        if mt >= cutoff:
            recent.append((mt, f))
    recent.sort(key=lambda x: x[0])  # 旧的先处理，新的覆盖旧的

    records, used = [], []
    for mt, f in recent:
        try:
            raw = json.load(open(f, 'r', encoding='utf-8'))
        except Exception as e:
            print(f"   ⚠️ 跳过 {os.path.basename(f)}: {e}")
            continue
        if isinstance(raw, dict):
            items = raw.get('jobs', [])
        elif isinstance(raw, list):
            items = raw
        else:
            items = []
        if items:
            for it in items:
                if isinstance(it, dict):
                    it['_src_file'] = os.path.basename(f)
                    it['_src_time'] = mt
                    records.append(it)
            used.append((os.path.basename(f), len(items)))
    return records, used


def build_dataframe():
    print("📂 正在加载最近的抓取数据…")
    jobs, job_files = _load_recent('boss_jobs_*.json')
    details, det_files = _load_recent('boss_details_*.json')
    print(f"   岗位列表文件 {len(job_files)} 个 / {len(jobs)} 条记录")
    print(f"   岗位详情文件 {len(det_files)} 个 / {len(details)} 条 JD")

    # --- 详情字典：job_id -> jd ---
    detail_map = {}
    for d in details:
        key = d.get('job_id') or d.get('job_link') or d.get('link')
        if not key:
            continue
        jd = (d.get('jd') or '').strip()
        skills = d.get('skill_tags') or []
        if isinstance(skills, str):
            try:
                skills = ast.literal_eval(skills)
            except Exception:
                skills = [s for s in re.split(r'[,|、\s]+', skills) if s]
        # 同 key 保留 JD 更长的那份（信息更全）
        prev = detail_map.get(key)
        if prev is None or len(jd) > len(prev.get('jd', '')):
            detail_map[key] = {'jd': jd, 'skill_tags': skills}

    # --- 岗位去重：job_id 为键，保留最新 ---
    merged = {}
    for j in jobs:
        key = j.get('job_id') or j.get('job_link')
        if not key:
            continue
        prev = merged.get(key)
        if prev is None or j['_src_time'] >= prev['_src_time']:
            merged[key] = j

    rows = []
    jd_hit = 0
    for key, j in merged.items():
        det = detail_map.get(key) or detail_map.get(j.get('job_link')) or {}
        jd = det.get('jd', '')
        if jd:
            jd_hit += 1
        row = dict(j)
        row['job_key'] = key
        row['jd'] = jd
        row['tech_list'] = det.get('skill_tags', []) or []
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"   去重后 {len(df)} 个岗位，其中 {jd_hit} 个成功合并到 JD "
          f"（{jd_hit / max(len(df), 1) * 100:.0f}%）")
    if jd_hit == 0:
        print("   ⚠️ 一个 JD 都没合并上！语义匹配会退回兜底分。")
        print("      检查：爬虫是否用了 --detail，详情是否落在 " + RESULT_DIR)
    return df


# ============================================================
# 第四步：匹配打分（每个岗位按它所属赛道的画像打分）
# ============================================================
W_TECH, W_EXP, W_EDU, W_SALARY, W_SEMANTIC = 30, 15, 10, 10, 35


def parse_experience(tags):
    if not isinstance(tags, str):
        return None
    for p in tags.split('|'):
        if '年' in p:
            return p.strip()
    return None


def parse_education(tags):
    if not isinstance(tags, str):
        return None
    for p in tags.split('|'):
        p = p.strip()
        if p in ['大专', '本科', '硕士', '博士', '学历不限']:
            return p
    return None


def parse_salary_mid(salary_str):
    """'15-25K' -> 20.0 ; '300-350元/天' -> 按 21.75 天折算成 K"""
    if not isinstance(salary_str, str):
        return None
    m = re.search(r'(\d+\.?\d*)-(\d+\.?\d*)\s*K', salary_str)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    m = re.search(r'(\d+\.?\d*)-(\d+\.?\d*)\s*元/天', salary_str)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2 * 21.75 / 1000
    m = re.search(r'(\d+\.?\d*)-(\d+\.?\d*)\s*元/月', salary_str)
    if m:
        return (float(m.group(1)) + float(m.group(2)) / 2) / 1000
    return None


def keyword_hits(text, keywords):
    low = str(text or '').lower()
    return sum(1 for k in keywords if k.lower() in low)


_AI_VECTORIZER = {}


def _zh_analyzer(text):
    """中文文本没法靠空格分词（整句就是一个 token），
    所以英文按词切 + 中文按字符 n-gram 切，保证中文之间也有真实的重合度。"""
    tokens = []
    for w in re.findall(r'[a-zA-Z][a-zA-Z0-9\+\#\.]*', str(text).lower()):
        if len(w) > 1:
            tokens.append(w)
    for chunk in re.findall(r'[\u4e00-\u9fff]+', str(text)):
        for n in (2, 3, 4):
            if len(chunk) >= n:
                tokens.extend(chunk[i:i + n] for i in range(len(chunk) - n + 1))
        if len(chunk) <= 4:
            tokens.append(chunk)
    return tokens


def semantic_score(profile, jd_text, profile_key):
    """词 + 字 n-gram 的 TF-IDF 余弦相似度，归一化到 0~1；模型按赛道缓存"""
    if not jd_text or len(jd_text) < 50:
        return None
    try:
        vec = _AI_VECTORIZER.get(profile_key)
        if vec is None:
            vec = TfidfVectorizer(analyzer=_zh_analyzer, max_features=2000,
                                  sublinear_tf=True, min_df=1)
            vec.fit([profile['personal_summary'], jd_text])
            _AI_VECTORIZER[profile_key] = vec
        m = vec.transform([profile['personal_summary'], jd_text])
        return float(cosine_similarity(m[0:1], m[1:2])[0][0])
    except Exception:
        return None


def score_row(row, profile):
    score, details = 0.0, {}

    # --- 技术栈 ---
    jd_tech = row.get('tech_list') or []
    jd_blob = f"{row.get('title', '')} {row.get('jd', '')} {' '.join(map(str, jd_tech))}"
    my_tech = profile['tech_skills']
    if jd_tech:
        matched = len(set(map(str.lower, my_tech)) & set(map(str.lower, map(str, jd_tech))))
        tech_score = min(matched / max(len(jd_tech), 1), 1.0) * W_TECH
    else:
        # 没有标签就从 JD 正文数关键词命中
        hits = keyword_hits(jd_blob, my_tech)
        tech_score = min(hits / 6.0, 1.0) * W_TECH
    score += tech_score
    details['技术栈'] = round(tech_score, 1)

    # --- 经验 ---
    exp_req = parse_experience(row.get('tags', ''))
    my_exp = profile['years_exp']
    if exp_req:
        if '不限' in exp_req or '在校' in exp_req or '应届' in exp_req:
            exp_score = W_EXP * 0.9
        else:
            nums = re.findall(r'(\d+)', exp_req)
            if len(nums) >= 2 and int(nums[0]) <= my_exp <= int(nums[1]):
                exp_score = W_EXP
            elif nums and my_exp < int(nums[0]):
                exp_score = W_EXP * 0.45 if int(nums[0]) <= 3 else W_EXP * 0.1
            else:
                exp_score = W_EXP * 0.8
    else:
        exp_score = W_EXP * 0.6
    score += exp_score
    details['经验'] = round(exp_score, 1)

    # --- 学历 ---
    edu_req = parse_education(row.get('tags', ''))
    rank = {'大专': 1, '本科': 2, '硕士': 3, '博士': 4}
    if edu_req:
        if edu_req == '学历不限':
            edu_score = W_EDU
        else:
            edu_score = W_EDU if rank.get(profile['education'], 0) >= rank.get(edu_req, 0) else 0.0
    else:
        edu_score = W_EDU * 0.7
    score += edu_score
    details['学历'] = round(edu_score, 1)

    # --- 薪资 ---
    sal_mid = parse_salary_mid(row.get('salary', ''))
    if sal_mid and profile['expected_salary'] > 0:
        if sal_mid >= profile['expected_salary']:
            salary_score = W_SALARY
        else:
            ratio = sal_mid / profile['expected_salary']
            salary_score = W_SALARY * (1.0 if ratio >= 0.9 else 0.6 if ratio >= 0.7 else 0.25)
    else:
        salary_score = W_SALARY * 0.3
    score += salary_score
    details['薪资'] = round(salary_score, 1)

    # --- 语义 ---
    sim = semantic_score(profile, row.get('jd', ''), profile['track'])
    if sim is None:
        sem_score = W_SEMANTIC * 0.3  # 没有 JD，只能给兜底分
    else:
        sem_score = min(sim * 2.2, 1.0) * W_SEMANTIC  # 余弦值普遍偏低，做一次放大
    score += sem_score
    details['语义'] = round(sem_score, 1)

    # --- 方向加减分 ---
    bonus = keyword_hits(jd_blob, profile['bonus_keywords'])
    penalty = keyword_hits(jd_blob, profile['penalty_keywords'])
    adj = min(bonus, 5) * 2.0 - min(penalty, 3) * 4.0
    score += adj
    details['方向'] = round(adj, 1)

    return round(max(score, 0.0), 2), '｜'.join(f"{k}{v}" for k, v in details.items())


# ============================================================
# 第五步：招呼语生成（按赛道路由项目，不再"默认讲爬虫"）
# ============================================================
def _fallback_greeting(row, profile, opening=None):
    """API 不可用时的降级模板。按岗位特征挑最贴的项目讲，
    开头用 _pick_opening 随机指定的那句（避免 40 条开头一模一样）。"""
    name = '马丁'
    title = row.get('title', '该岗位')
    blob = f"{title} {row.get('jd', '')}".lower()
    opening = opening or f"您好，我是{name}。"

    if profile['track'] == 'ai':
        if any(k in blob for k in ['agent', '智能体', 'mcp', '工具调用', 'workflow']):
            body = (f"我做过一个完整的 AI 应用闭环："
                    f"数据采集 + 匹配排序 + 调用大模型 API 生成内容，并用页面面板承载整个流程，"
                    f"对把模型能力接进实际业务这件事比较熟。想聊聊具体的技术方案，"
                    f"期待有机会进一步沟通。")
        elif any(k in blob for k in ['prompt', '图像', 'aigc', '风格', '生图']):
            body = (f"我在 Prompt 工程上做过比较系统的实践——"
                    f"用结构化模板控制多个维度，把生成结果的一致性从 60% 提到了 90% 以上，"
                    f"也沉淀了一套检查 SOP。这个方向和我的经验比较对口，"
                    f"希望能有机会进一步沟通。")
        else:
            body = (f"我基于 LangChain + Chroma 独立搭过一套 RAG 问答系统，"
                    f"从文档切分、向量化存储到语义检索和答案溯源整条链路都自己走过一遍，"
                    f"能把大模型能力落到具体业务里。这个岗位我很感兴趣，"
                    f"期待有机会进一步沟通。")
    else:
        # QA 赛道
        if any(k in blob for k in ['selenium', 'playwright', 'appium', 'ui自动化', '元素定位']):
            body = (f"我写过基于协议控制浏览器的自动化程序，"
                    f"包括定位页面元素、稳定抓取结构化数据，也踩过不少元素失效和超时的坑，"
                    f"对 UI 自动化的稳定性问题有实际体会。这个方向我想深入做，"
                    f"期待有机会进一步沟通。")
        elif any(k in blob for k in ['接口测试', 'api测试', 'postman', 'requests', '自动化测试']):
            body = (f"我日常用 Python 做接口调用和数据处理，"
                    f"熟悉 HTTP 协议和返回结构的校验，也独立写过完整的自动化脚本，"
                    f"包括异常分支的处理。这个岗位和我的方向比较一致，"
                    f"期待有机会进一步沟通。")
        else:
            body = (f"我做过一套无人值守的桌面自动化程序——"
                    f"用图像识别定位界面元素、模拟操作完成任务，主动处理了弹窗等异常情况，"
                    f"还打包成 exe 让非技术的同事直接用，上线后每天省下 30 分钟人工操作。"
                    f"这个岗位我很感兴趣，期待有机会进一步沟通。")

    return opening + body


# ---------- 开源链接：JD 明确要求时才附 ----------
OSS_URL = 'github.com/mading007/boss-zhipin-ai-assistant'

# JD 里出现这些，才说明对方真的想看你的代码/作品集。
#
# 注意：不能只匹配"开源""GitHub"两个词——实测大量岗位只是在说
# "熟悉开源软件的使用""采用开源的开发模式""跟踪开源工具"
# 或"会使用 GitHub 平台"（技能要求），这些都不该附链接。
# 所以要求「动作词 + 作品词」同时出现。
_OSS_ASK_PATTERN = re.compile(
    # ① 动作词 + 作品词（"请提供 GitHub 仓库""欢迎附作品截图"）
    r'(提供|附上|附|请附|请提供|给出|展示|欢迎附|注明)[^。；\n]{0,24}'
    r'(开源|GitHub|github|代码仓库|代码地址|作品集|作品|Demo|演示链接|演示录屏|项目链接|项目文档|博客)'
    r'|'
    # ② 作品词 + 明确指向作品的后缀（"作品集、项目文档、GitHub"里的"作品集"，"开源项目经验"）
    r'(作品集|开源项目|开源贡献|代码仓库|GitHub\s*仓库|Demo)[^。；\n]{0,16}'
    r'(优先|者优先|加分|可查|链接|地址|经验|经历|展示|公开|提交|附带)'
    r'|'
    # ③ "有...作品/开源项目" 这类要求，作品词必须带"可展示/可公开"等限定
    r'(有|具备)[^。；\n]{0,20}(开源项目|开源贡献|作品集|可公开展示|可展示的作品|github)',
    re.IGNORECASE,
)


def jd_wants_oss(jd_text):
    """判断 JD 是否明确要求提供开源项目 / 代码链接。
    用正则判定而不是交给模型，避免模型忽略这条规则（实测会忽略）。"""
    return bool(_OSS_ASK_PATTERN.search(str(jd_text or '')))


MAX_GREETING_LEN = 110


def _cap_length(text, limit=MAX_GREETING_LEN):
    """话术超长时，回退到 limit 内最后一个句末，避免硬切半句话。
    实测提示词里写多遍字数限制，模型照样写到 115 字，所以代码兜底。"""
    text = str(text or '').strip()
    if len(text) <= limit:
        return text
    head = text[:limit]
    # 优先在句末断开
    for ch in ['。', '！', '？', '；']:
        idx = head.rfind(ch)
        if idx >= limit * 0.5:
            return head[:idx + 1]
    return head.rstrip('，,、 ') + '。'


def _finalize(text, row, opening_unused=None):
    """统一收口：先决定要不要附仓库链接，再做长度兜底。
    顺序很重要——反过来的话，附加的链接会把总长度顶超上限。"""
    return _cap_length(_attach_oss(str(text or '').strip(), row))


def _attach_oss(greeting, row):
    """在'期待有机会进一步沟通'前插入仓库地址。已含链接则不重复加。"""
    if not jd_wants_oss(row.get('jd', '')):
        return greeting
    if OSS_URL in greeting or 'github.com' in greeting.lower():
        return greeting
    tail = '期待有机会进一步沟通'
    extra = f'我的项目代码在 {OSS_URL}，方便的话可以看一下。'
    if tail in greeting:
        return greeting.replace(tail, extra + tail, 1)
    return greeting.rstrip('。') + '。' + extra


# ---------- 招呼语开头：由 Python 随机指定，不交给模型选 ----------
# 实测把「开头要多样化」写进提示词时，模型会 100% 固定用其中一种（8/8 全一样），
# 所以改成代码随机挑选再通过 {opening} 注入提示词，模型只负责照抄。
#
# 同一批生成 40 条时，按顺序轮换这 3 种再打乱，保证分布均匀而非纯随机扎堆。
_OPENINGS = [
    '您好，我是马丁。',
    '您好，看到贵司在招{title}。',
    '您好，我是马丁，看到贵司的{title}岗位。',
]
# 只有 JD 明确写了校招/应届 时才允许用这一种
_OPENING_FRESH = '我是马丁，2026届电子信息工程专业，想投递{title}。'
_FRESH_PATTERN = re.compile(r'应届|校招|校园招聘|欢迎应届')
_opening_seq = []
_opening_lock = None


def _pick_opening(row):
    """按赛道路由 + 轮换 的方式挑一个开头，并填入真实岗位名。"""
    global _opening_lock, _opening_seq
    import threading
    if _opening_lock is None:
        _opening_lock = threading.Lock()

    title = str(row.get('title', '该岗位') or '该岗位').strip()
    opts = list(_OPENINGS)
    if _FRESH_PATTERN.search(str(row.get('jd', ''))):
        opts.append(_OPENING_FRESH)

    with _opening_lock:
        if not _opening_seq:
            # 新一批：轮换序列打乱，保证每种开头都被均匀用到
            import random
            n = len(opts)
            _opening_seq = [opts[i % n] for i in range(n * 4)]
            random.shuffle(_opening_seq)
        tpl = _opening_seq.pop(0)
    return tpl.replace('{title}', title)


def generate_greeting(row, profile):
    opening = _pick_opening(row)   # 先定开头，保证降级时也用同一句
    api_key = DEEPSEEK_CONFIG['api_key']
    if not api_key:
        return _finalize(_fallback_greeting(row, profile, opening), row)

    title = row.get('title', '该岗位')
    jd = row.get('jd', '') or ''
    jd_excerpt = jd[:1500]
    company = row.get('boss_name', '') or ''

    if profile['track'] == 'ai':
        project_rule = f"""2. 第二句讲你的主打项目：{profile['main_project']}（{profile['main_project_desc']}）。
   例外（仅当满足条件时才换）：
   - JD 里明确出现"Prompt""图像生成""AIGC图像""风格" → 换成「AI图像风格统一生成」项目（结构化Prompt模板多维控制，风格统一度60%→90%）
   - JD 里明确出现"Agent""智能体""工具调用""MCP""workflow" 且**没有**出现"RAG""知识库" → 换成「智能求职助手」项目（CDP采集 + 匹配算法 + 大模型API生成话术）
   注意：本赛道默认**不要**主动讲爬虫、数据采集。"""
    else:
        project_rule = f"""2. 第二句讲你的主打项目：{profile['main_project']}（{profile['main_project_desc']}）。
   重点体现：能独立设计自动化方案、会处理异常分支、有工程化意识（打包exe、非技术同事能用）。
   例外（仅当满足条件时才换）：
   - JD 里明确出现"UI自动化""Selenium""Playwright""Appium""元素定位" → 换成「智能求职助手」项目（基于CDP控制浏览器、定位页面元素、稳定采集结构化数据）
   - JD 里明确出现"接口测试""API测试""requests""Postman" → 重点讲你做过的API调用与数据结构化经验，并说明你熟悉HTTP协议与接口断言
   注意：本赛道不要主动讲大模型、RAG，除非 JD 明确要求。"""

    # 画像摘要只截前 320 字：塞满 4 个项目时模型会贪心地全写进去，导致话术超长
    profile_brief = profile['personal_summary'].strip()[:320]

    prompt = f"""请根据以下信息，为求职者撰写一段发给HR的打招呼语（80-95字，不要超过100字）：

【求职者背景】
{profile_brief}

【目标岗位】
公司：{company}
职位：{title}
岗位描述：{jd_excerpt}

【要求】
1. **开头必须用下面指定的这一句，原样照抄，不要改写、不要换别的说法：**
   「{opening}」
   **不要出现"2026届""应届生""本科毕业生"等身份标签**，除非 JD 里明确写了"欢迎应届生""校招""应届"。
{project_rule}
3. 第三句表达"能快速上手"这个岗位的工作，用"期待有机会进一步沟通"结尾。
4. 不要写"面试回复率提升至XX%"这类个人求职数据。
5. 不要出现"GitHub""开源""代码可查""代码仓库"等词，也不要写任何网址。
6. 不要罗列技能清单，要像人在说话。
7. 语气：正式、专业、诚恳，不要浮夸。
8. 直接输出招呼语正文，不要加任何额外说明或引号。总长度控制在 100 字以内。

招呼语："""

    try:
        resp = requests.post(
            DEEPSEEK_CONFIG['base_url'],
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            json={
                'model': DEEPSEEK_CONFIG['model'],
                'messages': [
                    {'role': 'system',
                     'content': '你是一位专业的求职顾问，擅长帮求职者写出真诚、有吸引力、突出项目能力的打招呼语。'},
                    {'role': 'user', 'content': prompt},
                ],
                'temperature': 0.8,
                'max_tokens': 300,
            },
            timeout=20,
            # DeepSeek 是国内服务，必须绕过系统代理。
            # 走代理会被绕到境外出口，导致 SSL 握手被中断（实测 SSLError）。
            proxies={'http': None, 'https': None},
        )
        if resp.status_code == 200:
            text = resp.json()['choices'][0]['message']['content'].strip()
            return _finalize(text.strip('"').strip(), row)
        print(f" [HTTP {resp.status_code} → 用模板]", end='')
        return _finalize(_fallback_greeting(row, profile, opening), row)
    except Exception as e:
        print(f" [异常 {type(e).__name__} → 用模板]", end='')
        return _finalize(_fallback_greeting(row, profile, opening), row)


# ============================================================
# 第六步：输出 HTML 面板
# ============================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI 投递助手 · __LABEL__</title>
<style>
    body { font-family: 'Segoe UI','Microsoft YaHei',sans-serif; background: #f5f7fa; padding: 20px; margin:0; }
    .header-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap:10px;}
    h2 { margin:0; }
    .track-badge { display:inline-block; padding:3px 10px; border-radius:20px; background:#2c3e50; color:#fff; font-size:12px; vertical-align:middle; margin-left:8px;}
    .btn-group button { margin-left: 8px; padding: 6px 14px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
    .btn-applied { background: #17a2b8; color: white; }
    .btn-ignored { background: #ffc107; color: #333; }
    .btn-reset { background: #dc3545; color: white; }
    .counter { background:#fff; border-radius:8px; padding:12px 16px; margin-bottom:12px; box-shadow:0 1px 4px rgba(0,0,0,.08); font-size:14px; }
    .counter b { font-size:18px; }
    .counter.ok b { color:#28a745; }
    .counter.warn b { color:#e67e22; }
    .counter.danger b { color:#dc3545; }
    .counter button { margin-left:12px; padding:4px 10px; border:1px solid #d2d2d7; background:#fff; border-radius:6px; cursor:pointer; font-size:12px;}
    table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #e0e4e8; font-size:14px; }
    th { background: #2c3e50; color: white; font-weight: 600; position:sticky; top:0;}
    tr:hover { background: #f1f5f9; }
    .match-score { font-weight: bold; color: #2c3e50; }
    .btn-copy { background: #3498db; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; margin-right: 3px; font-size:13px;}
    .btn-done { background: #28a745; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; margin-right: 3px; font-size:13px;}
    .btn-ignore { background: #ff9800; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; font-size:13px;}
    .toast { position: fixed; bottom: 20px; right: 20px; background: #2ecc71; color: white; padding: 10px 20px; border-radius: 6px; display: none; z-index: 999; }
    .greeting-preview { font-size: 12px; color: #777; max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: inline-block; vertical-align:middle;}
    .btn-done.marked { background: #6c757d; cursor: not-allowed; }
    .row-done td { background: #f0f0f0; color: #999; }
    .row-ignored td { background: #fff3cd; color: #856404; }
    .modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000; justify-content: center; align-items: center; }
    .modal-overlay.active { display: flex; }
    .modal-box { background: white; border-radius: 8px; max-width: 820px; width: 90%; max-height: 80vh; padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); overflow: auto; position: relative; }
    .modal-close { position: sticky; top: 0; float: right; background: #dc3545; color: white; border: none; border-radius: 4px; padding: 5px 15px; cursor: pointer; font-size: 16px; }
    .modal-box table { font-size: 13px; }
    .modal-box th { background: #e9ecef; color: #333; position:static;}
</style>
</head>
<body>
<div class="header-bar">
    <h2>📋 AI 匹配投递清单<span class="track-badge">__LABEL__</span></h2>
    <div class="btn-group">
        <button id="btnShowApplied" class="btn-applied">📋 已投递</button>
        <button id="btnShowIgnored" class="btn-ignored">📋 已忽略</button>
        <button id="btnResetApplied" class="btn-reset">🔄 重置</button>
    </div>
</div>

<div class="counter" id="counterBox">
    <span>今日已投：<b id="todayCount">0</b> / __DAILY_CAP__ 个</span>
    <span id="counterHint" style="color:#888;"></span>
    <button id="btnMinus">— 手滑，减一个</button>
</div>

<p><small>💡 「一键投递」= 复制招呼语 + 打开岗位页，<b>发送动作请你自己在页面上完成</b>。
建议点开 JD 先滑一遍、停 20~60 秒再发，不要连点。</small></p>
<p><small>💡 「仅复制」只复制话术，「不感兴趣」隐藏该岗位。</small></p>

<table id="jobTable">
<thead>
<tr>
    <th>匹配分</th><th>公司</th><th>职位</th><th>薪资</th><th>地区</th><th>方向分</th><th>操作</th>
</tr>
</thead>
<tbody>
__ROWS__
</tbody></table>
<div id="toast" class="toast"></div>

<div id="appliedModal" class="modal-overlay">
    <div class="modal-box">
        <button class="modal-close" id="modalCloseApplied">✕ 关闭</button>
        <h3>📋 已投递岗位列表</h3>
        <div id="appliedListContent"><p>加载中...</p></div>
    </div>
</div>

<div id="ignoredModal" class="modal-overlay">
    <div class="modal-box">
        <button class="modal-close" id="modalCloseIgnored">✕ 关闭</button>
        <h3>👎 已忽略岗位列表</h3>
        <div id="ignoredListContent"><p>加载中...</p></div>
    </div>
</div>

<script>
(function() {
    var TRACK = '__TRACK__';
    var DAY_CAP = __DAILY_CAP__;
    var KEY_APPLIED = 'appliedJobs_' + TRACK;
    var KEY_IGNORED = 'ignoredJobs_' + TRACK;
    var KEY_LOG     = 'applyLog_' + TRACK;

    function today() { return new Date().toISOString().slice(0,10); }
    function getApplied() { try { return JSON.parse(localStorage.getItem(KEY_APPLIED) || '[]'); } catch(e) { return []; } }
    function setApplied(l) { localStorage.setItem(KEY_APPLIED, JSON.stringify(l)); }
    function getIgnored() { try { return JSON.parse(localStorage.getItem(KEY_IGNORED) || '[]'); } catch(e) { return []; } }
    function setIgnored(l) { localStorage.setItem(KEY_IGNORED, JSON.stringify(l)); }

    function getLog() {
        try { var l = JSON.parse(localStorage.getItem(KEY_LOG) || '{}'); if (l.date !== today()) return {date: today(), count: 0}; return l; }
        catch(e) { return {date: today(), count: 0}; }
    }
    function setLog(l) { localStorage.setItem(KEY_LOG, JSON.stringify(l)); }

    function renderCounter() {
        var l = getLog();
        document.getElementById('todayCount').textContent = l.count;
        var box = document.getElementById('counterBox');
        var hint = document.getElementById('counterHint');
        box.className = 'counter';
        if (l.count <= 8)      { box.classList.add('ok');     hint.textContent = '节奏正常，保持分散投递'; }
        else if (l.count <= DAY_CAP) { box.classList.add('warn'); hint.textContent = '接近上限了，今天收工吧'; }
        else                   { box.classList.add('danger'); hint.textContent = '⚠️ 已超过安全阈值，继续高频容易触发风控'; }
    }

    function markApplied(jobId) {
        var list = getApplied();
        if (list.indexOf(jobId) === -1) { list.push(jobId); setApplied(list); }
        var row = document.querySelector('tr[data-job-id="' + CSS.escape(jobId) + '"]');
        if (row) {
            row.classList.add('row-done');
            var d = row.querySelector('.btn-done');
            if (d) { d.classList.add('marked'); d.textContent = '✅ 已投'; d.disabled = true; }
            var i = row.querySelector('.btn-ignore');
            if (i) { i.disabled = true; i.textContent = '已忽略'; }
        }
    }

    function markIgnored(jobId) {
        var list = getIgnored();
        if (list.indexOf(jobId) === -1) { list.push(jobId); setIgnored(list); }
        var row = document.querySelector('tr[data-job-id="' + CSS.escape(jobId) + '"]');
        if (row) {
            row.classList.add('row-ignored');
            row.style.display = 'none';
            var i = row.querySelector('.btn-ignore');
            if (i) { i.disabled = true; i.textContent = '已忽略'; }
            var d = row.querySelector('.btn-done');
            if (d) { d.disabled = true; }
        }
    }

    function hideDone() {
        var a = getApplied(), g = getIgnored();
        document.querySelectorAll('tr[data-job-id]').forEach(function(row) {
            var id = row.dataset.jobId;
            if (a.indexOf(id) !== -1 || g.indexOf(id) !== -1) row.style.display = 'none';
        });
    }

    function toast(msg) {
        var t = document.getElementById('toast');
        t.textContent = msg; t.style.display = 'block';
        setTimeout(function() { t.style.display = 'none'; }, 2500);
    }

    function copyText(text, cb) {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(cb).catch(function() { legacyCopy(text); cb(); });
        } else { legacyCopy(text); cb(); }
    }
    function legacyCopy(text) {
        var ta = document.createElement('textarea');
        ta.value = text; document.body.appendChild(ta); ta.select();
        try { document.execCommand('copy'); } catch(e) {}
        document.body.removeChild(ta);
    }

    function showModal(modalId, containerId, list) {
        var c = document.getElementById(containerId);
        if (!list.length) { c.innerHTML = '<p>暂无记录。</p>'; }
        else {
            var h = '<table><thead><tr><th>公司</th><th>职位</th><th>薪资</th><th>地区</th></tr></thead><tbody>';
            list.forEach(function(id) {
                var row = document.querySelector('tr[data-job-id="' + CSS.escape(id) + '"]');
                if (row) {
                    h += '<tr><td>' + (row.getAttribute('data-company')||'') + '</td><td>' +
                         (row.getAttribute('data-title')||'') + '</td><td>' +
                         (row.getAttribute('data-salary')||'') + '</td><td>' +
                         (row.getAttribute('data-district')||'') + '</td></tr>';
                } else { h += '<tr><td colspan="4">' + id + '</td></tr>'; }
            });
            c.innerHTML = h + '</tbody></table>';
        }
        document.getElementById(modalId).classList.add('active');
    }

    document.addEventListener('DOMContentLoaded', function() {
        document.querySelectorAll('.btn-copy').forEach(function(b) {
            b.addEventListener('click', function() { copyText(this.getAttribute('data-greeting'), function(){ toast('✅ 已复制招呼语'); }); });
        });
        document.querySelectorAll('.btn-done').forEach(function(b) {
            b.addEventListener('click', function() {
                if (this.classList.contains('marked')) return;
                var jobId = this.getAttribute('data-job-id');
                var link = this.getAttribute('data-link');
                var greeting = this.getAttribute('data-greeting');
                var log = getLog(); log.count += 1; setLog(log); renderCounter();
                copyText(greeting, function() { toast('✅ 招呼语已复制，链接已打开'); });
                markApplied(jobId);
                if (link && link !== '#') window.open(link, '_blank');
            });
        });
        document.querySelectorAll('.btn-ignore').forEach(function(b) {
            b.addEventListener('click', function() {
                if (this.disabled) return;
                var id = this.getAttribute('data-job-id');
                if (confirm('确定对此岗位不感兴趣吗？它将从列表中隐藏。')) markIgnored(id);
            });
        });
        document.getElementById('btnMinus').addEventListener('click', function() {
            var log = getLog(); log.count = Math.max(0, log.count - 1); setLog(log); renderCounter();
        });
        document.getElementById('btnShowApplied').addEventListener('click', function(){ showModal('appliedModal','appliedListContent', getApplied()); });
        document.getElementById('btnShowIgnored').addEventListener('click', function(){ showModal('ignoredModal','ignoredListContent', getIgnored()); });
        document.getElementById('modalCloseApplied').addEventListener('click', function(){ document.getElementById('appliedModal').classList.remove('active'); });
        document.getElementById('modalCloseIgnored').addEventListener('click', function(){ document.getElementById('ignoredModal').classList.remove('active'); });
        document.getElementById('appliedModal').addEventListener('click', function(e){ if (e.target === this) this.classList.remove('active'); });
        document.getElementById('ignoredModal').addEventListener('click', function(e){ if (e.target === this) this.classList.remove('active'); });
        document.getElementById('btnResetApplied').addEventListener('click', function() {
            if (confirm('确定清空本赛道的已投递/已忽略记录吗？（今日计数也会归零）')) {
                setApplied([]); setIgnored([]); setLog({date: today(), count: 0}); location.reload();
            }
        });
        renderCounter();
        hideDone();
    });
})();
</script>
</body>
</html>
"""

DAILY_CAP = 15        # 面板里的每日建议上限（纯提示，不锁死）
GREETING_LIMIT = 40   # 每个赛道最多给多少个岗位生成招呼语（要调大就改这里）


def build_html(df, profile):
    rows = []
    for _, r in df.head(GREETING_LIMIT).iterrows():
        boss = html.escape(str(r.get('boss_name', '未知公司')))
        title = html.escape(str(r.get('title', '未知职位')))
        salary = html.escape(str(r.get('salary', '面议')))
        district = html.escape(str(r.get('district', '') or ''))
        score = float(r['match_score'])
        link = str(r.get('job_link', '#') or '#')
        greeting = html.escape(str(r.get('greeting', '')), quote=True)
        preview = html.escape(str(r.get('greeting', ''))[:34])
        job_id = html.escape(str(r.get('job_key', link)), quote=True)
        bonus = r.get('detail_scores', '')
        m = re.search(r'方向(-?\d+\.?\d*)', str(bonus))
        bonus_txt = m.group(1) if m else ''

        rows.append(f"""
    <tr data-job-id="{job_id}" data-company="{boss}" data-title="{title}" data-salary="{salary}" data-district="{district}" data-link="{html.escape(link, quote=True)}">
        <td class="match-score">{score:.1f}</td>
        <td>{boss}</td>
        <td>{title}</td>
        <td>{salary}</td>
        <td>{district}</td>
        <td>{bonus_txt}</td>
        <td>
            <button class="btn-copy" data-greeting="{greeting}">📋 仅复制</button>
            <button class="btn-done" data-job-id="{job_id}" data-link="{html.escape(link, quote=True)}" data-greeting="{greeting}">📋 一键投递</button>
            <button class="btn-ignore" data-job-id="{job_id}">👎 不感兴趣</button>
            <div class="greeting-preview" title="{greeting}">{preview}…</div>
        </td>
    </tr>""")

    return (HTML_TEMPLATE
            .replace('__ROWS__', ''.join(rows))
            .replace('__LABEL__', profile['label'])
            .replace('__TRACK__', profile['track'])
            .replace('__DAILY_CAP__', str(DAILY_CAP)))


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 72)
    print("🤖 AI 投递助手 v2 · 双赛道版（AI应用 / 自动化测试）")
    print("=" * 72)

    if DEEPSEEK_CONFIG['api_key']:
        print("🔑 已检测到 DEEPSEEK_API_KEY，将调用大模型生成招呼语")
    else:
        print("⚠️  未设置 DEEPSEEK_API_KEY，招呼语将使用模板版")
        print("   设置方法：$env:DEEPSEEK_API_KEY=\"sk-xxxx\"  然后重跑本脚本")

    df = build_dataframe()
    if df.empty:
        print("❌ 没有可用岗位数据。请先跑爬虫：")
        print("   cd D:\\Desktop\\boss-zhipin-scraper-master")
        print("   uv run python scripts/boss_cdp_raw.py --keyword \"自动化测试\" --city 北京 --pages 3 --detail")
        return

    # 分类
    df['track'] = df.apply(classify_track, axis=1)
    unclassified = int(df['track'].isna().sum())
    print(f"\n🏷️  赛道分类：AI应用 {int((df['track'] == 'ai').sum())} 个 | "
          f"自动化测试 {int((df['track'] == 'qa').sum())} 个 | "
          f"未归类（跳过） {unclassified} 个")

    print(f"\n{'=' * 72}")
    for profile in ALL_PROFILES:
        sub = df[df['track'] == profile['track']].copy()
        label = profile['label']
        if sub.empty:
            print(f"\n⏭️  【{label}】没有匹配的岗位数据，跳过。")
            if profile['track'] == 'qa':
                print("     → 你还没抓过测试岗。跑一次就有了：")
                print("       cd D:\\Desktop\\boss-zhipin-scraper-master")
                print("       uv run python scripts/boss_cdp_raw.py --keyword \"自动化测试\" --city 北京 --pages 3 --detail")
                print("       uv run python scripts/boss_cdp_raw.py --keyword \"测试开发\" --city 北京 --pages 3 --detail")
            continue

        print(f"\n【{label}】{len(sub)} 个岗位，按匹配分排序…")

        # 标题硬排除（硬件/车载/算法测试等不匹配方向）
        ex_pat = profile.get('exclude_title_pattern')
        if ex_pat:
            hit = sub['title'].astype(str).str.contains(ex_pat, regex=True, na=False)
            if hit.any():
                dropped = sub.loc[hit, 'title'].tolist()
                print(f"   已按标题剔除 {len(dropped)} 个不匹配方向的岗位：")
                for t in dropped[:8]:
                    print(f"      ✂ {str(t)[:40]}")
                if len(dropped) > 8:
                    print(f"      … 另有 {len(dropped) - 8} 个")
                sub = sub[~hit]

        # 过滤硕士硬门槛（学历分直接归零的，没意义）
        sub = sub[~sub['tags'].astype(str).str.contains('硕士', na=False)]
        sub['district'] = sub['location'].astype(str).str.split('·').str[1].fillna('')
        scored = sub.apply(lambda r: pd.Series(score_row(r, profile), index=['match_score', 'detail_scores']), axis=1)
        sub['match_score'] = scored['match_score']
        sub['detail_scores'] = scored['detail_scores']
        sub = sub.sort_values('match_score', ascending=False)

        print(f"   生成招呼语中（{min(len(sub), GREETING_LIMIT)} 条）…")
        sub = sub.copy()
        sub['greeting'] = ''          # 必须先建列！否则下面的 .loc 赋值会被 pandas 静默丢弃
        for i, (idx, row) in enumerate(sub.head(GREETING_LIMIT).iterrows(), 1):
            print(f"   [{i:2d}/{min(len(sub), GREETING_LIMIT)}] {str(row.get('title', ''))[:22]}", end='')
            sub.loc[idx, 'greeting'] = generate_greeting(row, profile)
            print()

        # 只保留真的生成出招呼语的行，避免面板里出现复制不出去的空话术
        before = len(sub)
        sub = sub[sub['greeting'].astype(str).str.strip().ne('')]
        if len(sub) < before:
            print(f"   （{before - len(sub)} 个岗位超出话术生成上限，未进入面板）")

        # 输出 CSV
        cols = ['boss_name', 'title', 'salary', 'district', 'company_scale',
                'company_industry', 'job_link', 'match_score', 'detail_scores', 'greeting']
        cols = [c for c in cols if c in sub.columns]
        csv_name = f"apply_ready_list_{profile['track']}.csv"
        csv_path = os.path.join(SCRIPT_DIR, csv_name)
        sub[cols].to_csv(csv_path, index=False, encoding='utf-8-sig')

        # 输出 HTML
        html_name = f"apply_helper_{profile['track']}.html"
        html_path = os.path.join(SCRIPT_DIR, html_name)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(build_html(sub, profile))

        print(f"\n   📋 TOP 8 预览：")
        for i, (_, row) in enumerate(sub.head(8).iterrows(), 1):
            print(f"   {i:2d}. {row['match_score']:5.1f} | {str(row.get('boss_name', ''))[:16]:16s} | "
                  f"{str(row.get('title', ''))[:24]:24s} | {row.get('salary', '')}")
            print(f"       [{row.get('detail_scores', '')}]")
            print(f"       💬 {str(row.get('greeting', ''))[:70]}")
        print(f"\n   ✅ CSV  → {csv_path}")
        print(f"   ✅ 面板 → {html_path}")

    print(f"\n{'=' * 72}")
    print("💡 手动投递建议（重要，直接关系账号安全）：")
    print("   1. 一天投 8~15 个，分散在上午/下午，别一次性点完")
    print("   2. 点开后先滑一遍 JD，停 20~60 秒再发，不要连点")
    print("   3. 每个招呼语改一下首句，提到对方公司/业务的具体点")
    print("   4. 在 9:30-18:00 之间操作，别凌晨批量")
    print("=" * 72)


if __name__ == '__main__':
    main()
