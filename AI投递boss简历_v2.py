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
1. 知识库问答（8月，已开源）：基于 LangChain + Chroma + BGE Embedding 独立开发，
   把 PDF 等文档解析、切分、向量化后做语义检索，回答带原文出处，能直接接进业务系统；
2. 智能求职助手（9月）：Chrome DevTools Protocol 采集结构化数据，TF-IDF + 余弦相似度做匹配排序，
   调用大模型 API 生成定制内容，并用网页面板承载整个流程，形成完整 AI 应用闭环；
3. AI图像批量生成（6月）：用结构化 Prompt 模板控制光源/色调/景别/场景多个维度，
   把出图一致性从 60% 提升至 90% 以上，并沉淀成可复用的检查清单；
4. 金融终端自动化导出系统（6月）：图像识别定位 + 模拟键鼠操作，实现无人值守定时导出与邮件投递，
   打包成 exe 供非技术同事使用，上线后每天省下 30 分钟人工操作。

技术栈：Python、LangChain、Chroma、BGE Embedding、Prompt Engineering、
大模型 API 调用（DeepSeek/ChatGLM）、Pandas、MySQL、自动化脚本。
""",
    # 招呼语默认讲哪个项目（按 JD 关键词再切换）
    'main_project': '智能求职助手',
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
3. 知识库问答（8月，已开源）：LangChain + Chroma，搭过检索增强问答的完整链路，
   并对切分粒度与召回效果逐环节做过对比验证。

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


# ---------- 公司可信度 / HR 活跃度：两个影响「投了有没有用」的因素 ----------

# BOSS 上把公司名显示成「某大型互联网公司」的，基本都是猎头或外包代招，
# 点进去看不到真公司、投了也不知道去向。这类降权。
# 注意：只匹配「某 + 规模/性质 + 行业/公司」这种匿名化写法，
# 不能只看到「某」就判——真实公司名里也可能带这个字。
_ANON_COMPANY_PATTERN = re.compile(
    r'^某|'
    r'某(大型|中型|小型|知名|上市|头部|独角兽)?[^，,]{0,10}(公司|集团|企业|银行|机构|平台)|'
    # 没有「某」但同样匿名化的写法
    r'^(知名|著名|大型|头部|一线|上市|独角兽)[^，,]{0,8}公司$|'
    r'^(知名公司|保密公司|匿名公司)$'
)
_HEADHUNTER_PATTERN = re.compile(r'人力资源|人才服务|猎头|劳务|外包|外派|派遣|代招|招聘服务')


def company_credibility(name):
    """返回 (加减分, 标签)。匿名或猎头外包岗降权。"""
    s = str(name or '').strip()
    if not s:
        return 0.0, ''
    if _ANON_COMPANY_PATTERN.search(s):
        return -8.0, '匿名/猎头'
    if _HEADHUNTER_PATTERN.search(s):
        return -5.0, '人力/外包'
    return 0.0, ''


# HR 活跃状态 -> (加减分, 展示标签)
# BOSS 机制：HR 不在线时招呼语不会推送到他手机，会躺在后台等他上线。
# 所以「在线」的岗位才值得优先投。
def hr_activity(status):
    s = str(status or '').strip()
    if not s:
        return 0.0, ''

    # 先排掉带时间前缀的（「8小时前在线」「3天前在线」），否则会被当成"在线"
    m = re.search(r'(\d+)\s*小时前', s)
    if m:
        h = int(m.group(1))
        return (2.0, f'{h}小时前') if h <= 12 else (-1.0, f'{h}小时前')
    m = re.search(r'(\d+)\s*天前', s)
    if m:
        return -4.0, f'{m.group(1)}天前'
    m = re.search(r'(\d+)\s*分钟前', s)
    if m:
        return 5.0, f'{m.group(1)}分钟前'

    # 无时间前缀的即时状态
    if '刚刚' in s:
        return 5.0, '刚刚活跃'
    if '当前在线' in s or s == '在线' or '在线' in s:
        return 5.0, '在线'
    return 0.0, s[:8]


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

    # --- 公司可信度：匿名/猎头/外包岗降权 ---
    cred_adj, cred_tag = company_credibility(row.get('boss_name', ''))
    score += cred_adj
    details['公司'] = cred_adj

    # --- HR 活跃度：在线优先（BOSS 上 HR 离线时招呼语不会推送）---
    act_adj, act_tag = hr_activity(row.get('boss_active_status', ''))
    score += act_adj
    details['活跃'] = act_adj

    # 注意：这些标签必须由调用方写回 DataFrame 列。
    # 这里只写 row['xxx'] 是改 Series 副本，不会创建列（历史上已踩坑三次）。
    return {
        'match_score': round(max(score, 0.0), 2),
        'detail_scores': '｜'.join(f"{k}{v}" for k, v in details.items()),
        'company_tag': cred_tag,
        'active_tag': act_tag,
    }


# ============================================================
# 第五步：招呼语生成（按赛道路由项目，不再"默认讲爬虫"）
# ============================================================
def _fallback_greeting(row, profile, opening=None):
    """API 不可用时的降级模板。开头与项目都沿用 Python 侧的选定结果，
    避免 API 挂掉时 40 条话术开头和正文全都一样。"""
    name = '马丁'
    opening = opening or f"您好，我是{name}。"
    proj, _ = _pick_project(row, profile)

    if proj:
        tools_txt = '、'.join(proj.get('tools', []))
        body = (f"我用 {tools_txt} {proj['desc']}。"
                f"这个岗位我很感兴趣，期待有机会进一步沟通。")
    else:
        body = "我做过与这个岗位方向相关的完整项目，能独立把事情落地。期待有机会进一步沟通。"

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
_CLOSING = '期待有机会进一步沟通。'


def _trim_to_sentence(text, budget):
    """把 text 截到不超过 budget 字，且尽量停在一个完整句末。

    注意：补句号时是多出一个字符的，所以补之前要先腾出位置，
    否则会稳定超出 budget 一个字（曾实测所有超长都刚好是 111）。
    """
    if len(text) <= budget:
        return text

    cut = text[:budget]
    for ch in ['。', '！', '？', '；']:
        idx = cut.rfind(ch)
        if idx >= budget * 0.5:
            return cut[:idx + 1]          # 已经是句末，直接用

    # 没有可用句末：去掉尾部标点后补一个句号，注意别越界
    core = cut[:budget - 1].rstrip('，,、；; ')
    return core + '。'


def _cap_length(text, limit=MAX_GREETING_LEN):
    """话术超长时的兜底。

    模型无视字数限制（实测仍会写到 115 字），所以代码兜底。两个要求：
      1. 不出现「…能直接。」这种半句话 —— 尽量停在句末
      2. 保住收尾句「期待有机会进一步沟通。」—— 但绝不能因此超过 limit

    做法：先切掉已有的收尾句，按「limit − 收尾句长度」给正文留预算，
    再把收尾句接回去；正文里本来就没有收尾句时，不硬加（避免话术变僵）。
    """
    text = str(text or '').strip()
    if len(text) <= limit:
        return text

    has_closing = _CLOSING in text
    if has_closing:
        body = text[:text.rfind(_CLOSING)]
        body = _trim_to_sentence(body, limit - len(_CLOSING))
        return body + _CLOSING

    return _trim_to_sentence(text, limit)


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

# ---------- 主打项目：同样由 Python 决定，不交给模型 ----------
# 每个项目配一组「JD 命中词」。命中最多者优先；都不命中时按顺序轮换，
# 避免 40 条话术全都讲同一个项目（实测同一批 10 条全讲金融终端，很像群发）。
# 每个项目配三样东西：
#   name  项目内部名（也用于降级模板）
#   tools 可被念出来的技术名 —— 招呼语要**先亮这些词**，HR 扫一眼就能命中关键词
#   desc  做出来能干嘛（业务价值），不要写成「我怎么实现的」
#   keys  命中词，用于判断这个岗位该讲哪个项目
#
# 命名注意：不要把项目叫成「XX论文系统」——「论文」两字会让人一眼判定是课程作业。
_PROJECTS = {
    'ai': [
        {
            'name': '智能求职助手',
            'tools': ['Chrome DevTools Protocol', 'TF-IDF', '大模型 API'],
            'desc': '做过一个从数据采集到内容生成的全链路 AI 应用：自动采集结构化数据、'
                    '用 TF-IDF 与余弦相似度做匹配排序、再调大模型 API 生成定制内容，'
                    '并用网页面板把整条流程承接起来',
            'keys': ['agent', '智能体', 'mcp', '工具调用', 'workflow', '编排',
                     'function calling', '全栈', '端到端', '闭环'],
        },
        {
            'name': '知识库问答',
            'tools': ['LangChain', 'Chroma', 'BGE Embedding'],
            'desc': '用这套做过一套面向企业内部文档的问答服务：把 PDF 等非结构化文档'
                    '解析、切分、向量化后做语义检索，回答带原文出处，能直接接进业务系统',
            'keys': ['rag', 'langchain', 'chroma', '向量', '知识库', '语义检索',
                     'embedding', '召回', '检索增强', '问答'],
        },
        {
            'name': 'AI图像批量生成',
            'tools': ['Prompt Engineering', '结构化模板'],
            'desc': '用结构化 Prompt 模板控制光源、色调、景别、场景等维度做批量生成，'
                    '把出图一致性从六成提到九成以上，并沉淀成可复用的检查清单',
            'keys': ['prompt', '提示词', 'aigc', '图像', '生图', '风格', '多模态', '文生图'],
        },
    ],
    'qa': [
        {
            'name': '金融终端自动化导出系统',
            'tools': ['Python', '图像识别定位', '模拟键鼠操作'],
            'desc': '做过一套无人值守的桌面自动化程序：自动完成导出、加时间戳归档、'
                    '按日期合并汇总并邮件发送，主动处理了弹窗等异常分支，'
                    '打包成 exe 让非技术同事直接双击使用，上线后每天省下 30 分钟人工操作',
            'keys': ['自动化脚本', '定时任务', '无人值守', 'windows', '桌面',
                     'exe', '批量处理', '运维'],
        },
        {
            'name': '智能求职助手',
            'tools': ['Chrome DevTools Protocol', '元素定位', '结构化采集'],
            'desc': '用 CDP 控制浏览器稳定采集数据结构化落库，并针对元素失效、页面加载'
                    '超时等情况做了重试与降级处理，保证长时间运行不中断',
            'keys': ['selenium', 'playwright', 'appium', 'ui自动化', '元素定位',
                     '爬虫', '抓取', 'cdp', 'chromedriver', 'web自动化'],
        },
        {
            'name': '接口联调与数据处理',
            'tools': ['Python 接口调用', 'HTTP 协议', 'JSON 结构校验'],
            'desc': '日常用 Python 做接口联调与数据清洗，能根据返回结构写字段校验与'
                    '异常分支处理，把接口数据整理成可直接分析的表格',
            # 只放「接口测试」这类专有说法。
            # 早期误放了 'http'、'requests' 等泛词，导致 50% 测试岗都被判成这一项。
            'keys': ['接口测试', '接口自动化', 'api测试', 'api自动化',
                     'postman', 'jmeter', '接口联调'],
        },
        {
            'name': '知识库问答',
            'tools': ['LangChain', 'Chroma', 'RAG 链路验证'],
            'desc': '搭过检索增强问答的完整链路，并对切分粒度、召回效果逐环节做过对比验证，'
                    '能为 AI 类产品的效果评估提供可复用的方法',
            # 这里只保留 AI 测试专用词。原先放了 '大模型''ai应用' 等泛词，
            # 几乎每个 AI 岗都会命中，把本该分给其它项目的岗位全抢走了。
            'keys': ['ai测试', '模型评测', '大模型测试', 'rag', 'llm'],
        },
    ],
}
_project_cursor = {}

# 关键词全都没命中时，默认讲哪个项目。
# QA 用金融终端（测试岗硬通货：无人值守 + 异常处理 + 打包交付）；
# AI 用智能求职助手（技术链路最全：采集 + 匹配算法 + 大模型 API + 前端面板）。
DEFAULT_PROJECT = {
    'ai': '智能求职助手',
    'qa': '金融终端自动化导出系统',
}


def _pick_project(row, profile):
    """按 JD 关键词选主打项目；都不命中时用默认项目，并按顺序轮换保证分布。"""
    global _project_cursor
    jd = str(row.get('jd', '')).lower()   # 只看 JD，标题里带自己项目名会造成自匹配
    specs = _PROJECTS.get(profile['track'], [])
    if not specs:
        return None, 0

    scored = []
    for i, p in enumerate(specs):
        hits = sum(1 for k in p['keys'] if k in jd)
        scored.append((hits, i, p))

    best_hits = max((s[0] for s in scored), default=0)
    if best_hits > 0:
        # 命中最多者优先；并列时轮换，避免永远只选第一个
        tied = [s for s in scored if s[0] == best_hits]
        cur = _project_cursor.get(profile['track'], 0)
        chosen = tied[cur % len(tied)]
        _project_cursor[profile['track']] = cur + 1
        return chosen[2], best_hits

    # 无命中：用默认项目（找得到就用，找不到退回第一个）
    default_name = DEFAULT_PROJECT.get(profile['track'], specs[0]['name'])
    for p in specs:
        if p['name'] == default_name:
            return p, 0
    return specs[0], 0


# ============================================================
# JD 技术词提取：把岗位真正在意的技术名挑出来，提示词里点名让 AI 呼应
# ============================================================
# 词表按「具体 → 泛化」排列。越靠前的词越能体现岗位的技术栈，
# 提取时优先取它们，避免把「沟通能力」「团队协作」这类放进招呼语。
_TECH_VOCAB = [
    # 语言 / 数据
    'Python', 'Java', 'Go', 'Golang', 'C++', 'SQL', 'MySQL', 'PostgreSQL',
    'Redis', 'MongoDB', 'Elasticsearch', 'Pandas', 'NumPy',
    # 大模型 / RAG
    'LangChain', 'LlamaIndex', 'Chroma', 'Milvus', 'Faiss', '向量数据库',
    'RAG', 'Embedding', 'Prompt', '提示词', '微调', 'Fine-tuning',
    '大模型', 'LLM', 'GPT', 'ChatGLM', 'Qwen', 'DeepSeek',
    # Agent / 工作流
    'Agent', '智能体', 'MCP', 'Function Calling', '工作流', 'Workflow',
    'Dify', 'Coze', 'LangGraph', 'AutoGen',
    # 工程 / 部署
    'Docker', 'Kubernetes', 'K8s', 'Linux', 'Git', 'CI/CD', 'Jenkins',
    'FastAPI', 'Flask', 'Django', 'Spring', 'Vue', 'React',
    # 测试向
    'pytest', 'unittest', 'Selenium', 'Playwright', 'Appium', 'Postman',
    'JMeter', '接口测试', '自动化测试', '测试用例', '性能测试',
    'UI自动化', '接口自动化', '测试开发', '缺陷',
    # 数据 / 分析
    'Excel', 'Tableau', 'Power BI', '数据分析', '数据清洗', '爬虫',
    '数据采集', '数据可视化',
]

_TECH_LOWER = [(t, t.lower()) for t in _TECH_VOCAB]


def extract_jd_tech(jd_text, limit=6):
    """从 JD 里挑出最该被呼应的技术词。

    返回按出现顺序排列的关键词列表（保持词表的优先级顺序，越具体的越靠前）。
    注意：匹配时要求词边界或原样出现，避免 'go' 命中 'google' 这类误判。
    """
    jd = str(jd_text or '')
    if not jd:
        return []
    low = jd.lower()
    hits = []
    for orig, lw in _TECH_LOWER:
        if len(lw) <= 3 and lw.isascii():
            # 短英文词要求非字母边界，防止 go/git 之类误命中
            if re.search(r'(?<![a-zA-Z])' + re.escape(lw) + r'(?![a-zA-Z])', low):
                hits.append(orig)
        elif lw in low:
            hits.append(orig)
        if len(hits) >= limit:
            break
    return hits


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

    # 主打项目由 Python 选定（按 JD 关键词优先，无命中则走默认），模型只负责把它讲自然
    proj, proj_hits = _pick_project(row, profile)

    # 从 JD 里挑出这个岗位真正在意的技术词，点名让 AI 在话术里呼应
    jd_tech = extract_jd_tech(jd)

    if proj:
        tools_txt = '、'.join(proj.get('tools', []))
        project_rule = (
            f"2. **第二句讲下面这个指定经历，不要换成你背景里别的经历：**\n"
            f"   可选技术名：{tools_txt}\n"
            f"   做出来能干嘛：{proj['desc']}\n"
            f"   要求：**从「可选技术名」里挑 1 个念出来就够，不要三个都念**"
            f"（挑跟这个岗位最相关的那个）。重点是说清拿它做成了什么，"
            f"不要报项目名，也不要把技术名堆成一串。"
        )
    else:
        project_rule = "2. 第二句讲一个你最相关的项目经历，先说用了什么技术，再说做成了什么。"

    # JD 技术词呼应要求：只在真的提取到词时才加，避免空规则干扰模型
    if jd_tech:
        tech_rule = (
            f"3. 从上面「岗位描述」里挑 1-2 个最具体的技术点来呼应"
            f"（这个岗位明确提到了：{'、'.join(jd_tech)}），"
            f"说明你正好做过对应的事。**只挑 1-2 个，不要全念一遍，也不要罗列。**"
        )
    else:
        tech_rule = "3. 如果岗位描述里提到了具体技术，挑 1-2 个最相关的进行呼应，不要罗列。"

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
{tech_rule}
4. 结尾用"期待有机会进一步沟通"。
5. 不要写"面试回复率提升至XX%"这类个人求职数据。
6. 不要出现"GitHub""开源""代码可查""代码仓库"等词，也不要写任何网址。
7. **不要列成"我熟悉A、B、C"这种清单**——技术名要揉进句子里说，像在讲自己做过的一件事。
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

    /* 操作列：按钮一行 + 完整招呼语预览 */
    td.op-cell { min-width: 340px; }
    .op-row { display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 6px; }
    .greeting-text {
        font-size: 11.5px; line-height: 1.65; color: #5a5a5f;
        background: #f7f8fa; border-left: 3px solid #c7d2e0;
        border-radius: 4px; padding: 6px 9px;
        max-width: 340px; white-space: normal; word-break: break-word;
        user-select: text; cursor: text;
    }
    .row-done .greeting-text { background: #f0f0f0; border-left-color: #d0d0d0; color: #aaa; }
    .row-ignored .greeting-text { background: #fff8e1; border-left-color: #e0c080; }
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
    <th>匹配分</th><th>HR状态</th><th>公司</th><th>职位</th><th>薪资</th><th>地区</th><th>方向分</th><th>操作</th>
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

        # HR 活跃状态 + 公司可信度标签
        act_tag = html.escape(str(r.get('active_tag', '') or ''))
        comp_tag = html.escape(str(r.get('company_tag', '') or ''))
        if '在线' in act_tag:
            act_html = f'<span style="color:#28a745;font-weight:600">● {act_tag}</span>'
        elif act_tag.endswith('天前'):
            act_html = f'<span style="color:#dc3545">{act_tag}</span>'
        elif act_tag.endswith('小时前'):
            act_html = f'<span style="color:#e67e22">{act_tag}</span>'
        else:
            act_html = f'<span style="color:#bbb">未知</span>'
        if comp_tag:
            act_html += f'<br><span style="color:#dc3545;font-size:11px">⚠ {comp_tag}</span>'

        rows.append(f"""
    <tr data-job-id="{job_id}" data-company="{boss}" data-title="{title}" data-salary="{salary}" data-district="{district}" data-link="{html.escape(link, quote=True)}">
        <td class="match-score">{score:.1f}</td>
        <td>{act_html}</td>
        <td>{boss}</td>
        <td>{title}</td>
        <td>{salary}</td>
        <td>{district}</td>
        <td>{bonus_txt}</td>
        <td class="op-cell">
            <div class="op-row">
                <button class="btn-copy" data-greeting="{greeting}">📋 仅复制</button>
                <button class="btn-done" data-job-id="{job_id}" data-link="{html.escape(link, quote=True)}" data-greeting="{greeting}">📋 一键投递</button>
                <button class="btn-ignore" data-job-id="{job_id}">👎 不感兴趣</button>
            </div>
            <div class="greeting-text">{greeting}</div>
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
        scored = sub.apply(lambda r: pd.Series(score_row(r, profile)), axis=1)
        for col in ['match_score', 'detail_scores', 'company_tag', 'active_tag']:
            sub[col] = scored[col]
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
        cols = ['boss_name', 'active_tag', 'company_tag', 'title', 'salary', 'district',
                'company_scale', 'company_industry', 'job_link', 'match_score',
                'detail_scores', 'greeting']
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
