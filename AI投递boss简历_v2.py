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
                         'NLP算法', 'CV算法', '推荐算法', '硕士',
                         # 运营类不是不能干，但跟开发技能不对口，降权即可
                         '运营', '客服', '销售', '市场'],
    # 这些岗位即使 JD 里堆满 AI 关键词也不投——技能完全不匹配，纯浪费
    # 注意：只匹配「标题」。「京东/京东科技」里含"咨询"属误伤，故用词边界更严的词。
    'exclude_title_pattern': r'讲师|培训师|课程|营销|渠道|代理|推广|主播|带货|地推',
    # 标题再排除一层：PHP 后端、算法研究、机器视觉 —— 核心技术栈对不上
    'exclude_title_extra': r'\bPHP\b|算法工程师|机器视觉|计算机视觉|深度学习|模型训练',
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
    # 标题再排除一层：硬件/嵌入式/车载/算法研究 —— 核心技术栈对不上
    'exclude_title_extra': (
        r'嵌入式|单片机|硬件|射频|天线|PCB|BSP|'
        r'算法工程师|机器视觉|计算机视觉|深度学习|模型训练'
    ),
    'personal_summary': """
我是马丁，2026届电子信息工程本科毕业生，可立即到岗，期望城市北京。

核心项目：
1. 金融终端自动化导出系统（6月）：独立设计并实现无人值守的桌面自动化程序。
   针对交易客户端界面复杂、每日导出覆盖旧文件、人工重复操作繁琐的痛点，
   通过图像识别定位界面元素、模拟键鼠操作，实现定时任务；
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


# BOSS 详情页的 JD 正文尾部会混入页面底部的公司资质信息，例如：
#   「…认证资质 人力资源服务许可证 劳务派遣经营许可证 营业执照 …」
# 这段文字对所有公司都一样，留着会污染关键词匹配与语义打分
# （实测导致「劳务派遣」这类规则误命中所有岗位）。
_PAGE_FOOTER = re.compile(
    r'认证资质|人力资源服务许可证|劳务派遣经营许可证|朝阳区人社局监督电话'
)


def _strip_page_footer(jd):
    """切掉 JD 尾部的页面 footer 污染。"""
    if not jd:
        return jd
    m = _PAGE_FOOTER.search(jd)
    return jd[:m.start()].rstrip() if m else jd


# ============================================================
# 全局硬排除：命中即不投，不参与打分
# ============================================================
# 设计原则：只在有「硬证据」时排除，不因为「提到了某个语言」就排除。
# 曾经想按「不会的语言」过滤，但实测会误杀大量岗位——
# 例：FunPlus 写「熟悉 Python/Go/Java 中至少一种」，你会 Python 就够；
#     纬致写「至少掌握一种：Python/Java」，同理。
# 所以只在「核心技术栈明显不对口」时才剔。
_GLOBAL_EXCLUDE = {
    # 经验硬门槛：5 年以上。你刚毕业，这个门槛是真实存在的墙。
    # 注：3 年以上不硬剔（有些岗写 3 年但接受优秀应届），只靠打分降权。
    '经验要求 5 年以上': r'5\s*年(?:及)?以上|五年以上',
    # JD 明说是外包/驻场岗（本人明确不接受外包）
    '明说外包/驻场': r'驻场|外包岗|外包公司|第三方签约|派遣制|此为外包',
    # 外包/人力派遣公司。只列确实的派遣商；
    # 慧博云通、京北方、纬致等正规公司不在此列（曾误伤，已移出）
    '外包/派遣公司': (
        r'外企德科|柯莱特|中软国际|软通动力|法本信息|博彦科技|'
        r'文思海辉|中电金信|人瑞人才|上海佩航|联合永道|联想利泰|联想弘扬'
    ),
    # PHP 专属框架（出现即说明必须会 PHP，这是硬证据，不是语言列表）
    'PHP 专属框架': r'Laravel|Lumen|Yaf|ThinkPHP|Symfony',
}


def _exclude_reason(row, profile):
    """返回该岗位被硬排除的原因列表；空列表表示保留。"""
    title = str(row.get('title', ''))
    company = str(row.get('boss_name', '') or row.get('company', ''))
    jd = str(row.get('jd', ''))

    why = []
    # 标题层：讲师/营销/硬件/算法等方向不符
    for key, label in (('exclude_title_pattern', '标题-方向不符'),
                       ('exclude_title_extra', '标题-技术栈不符')):
        pat = profile.get(key)
        if pat and re.search(pat, title, re.I):
            why.append(label)
    # 全局层
    for name, pat in _GLOBAL_EXCLUDE.items():
        target = company if name == '外包/派遣公司' else (title + ' ' + jd)
        if re.search(pat, target, re.I):
            why.append(name)
    return why


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
        row['jd'] = _strip_page_footer(jd)
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
    """API 不可用时的降级模板。

    开头与能力都沿用 Python 侧的选定结果，避免 API 挂掉时
    几十条话术的开头和正文全都雷同。
    """
    name = '马丁'
    opening = opening or f"您好，我是{name}。"
    caps, _ = pick_capabilities(row, limit=2)

    if caps:
        body = '；'.join(cap_say(c) for c in caps) + "。期待有机会进一步沟通。"
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
    """把 text 截到不超过 budget 字，且停在读得通的收尾处。

    三个坑（都实测踩过）：
      · 补句号会多占一个字符，补之前要先腾位置，否则稳定超出上限 1 个字。
      · 截断点可能落在英文单词中间，出现过「基于 LangChain 和 Chrom。」
        —— Chroma 被切成 Chrom，像打错字。
      · 只退到「词边界」还不够：退过头会剩下「…答案，基于。」这种悬空收尾。
        所以**优先在中文标点处收尾**（「…答案。」读着完整），
        中文标点离得太远时，才退到词边界。
    """
    if len(text) <= budget:
        return text

    cut = text[:budget]
    # ① 优先：句末标点
    for ch in ['。', '！', '？', '；']:
        idx = cut.rfind(ch)
        if idx >= budget * 0.5:
            return cut[:idx + 1]
    # ② 次选：中文逗号/顿号（补句号即可成句）
    for ch in ['，', '、']:
        idx = cut.rfind(ch)
        if idx >= budget * 0.6:
            return cut[:idx] + '。'

    # ③ 兜底：退到词边界（避免切碎英文单词），再补句号
    core = cut[:budget - 1]
    # 截断点前的空白也一并去掉，否则会生成「it 。」这种带空格再接句号的情况
    core = core.rstrip(' \t')
    m = re.search(r'[A-Za-z0-9][A-Za-z0-9.+\-_/ ]*$', core)
    if m and m.start() > 0:
        core = core[:m.start()]
    # 退完之后如果中文标点还在附近，用它收尾更自然
    for ch in ['，', '、']:
        idx = core.rfind(ch)
        if idx >= len(core) * 0.6:
            return core[:idx] + '。'
    core = core.rstrip(' \t，,、；;:：')
    if not core:
        core = cut[:budget - 1].rstrip(' \t，,、；;:：')
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


def _ensure_closing(text, limit=MAX_GREETING_LEN):
    """保证话术以完整句子收尾。

    模型偶尔会输出没写完的话（实测出现过「…能直接接进业务；」这种
    以分号结尾、话没说完的情况），直接发出去很难看。
    这里做两层兜底：

    1. 收尾句前面必须是句号。模型有时写「…流程；期待有机会进一步沟通。」
       或「…流程，期待…」，分号/逗号后接独立句是病句。
    2. 结尾不是句号/问号/叹号时，补收尾句（放得下时）。

    标点用正则统一处理（含半角与中间夹杂空格的情况），不硬编码全角字符——
    早先写成 t.replace('；'+CLOSING, ...) 时，模型偶尔输出半角分号或带空格
    的变体就漏掉了（实测 40 条里漏了 5 条）。
    """
    t = str(text or '').strip()
    if not t:
        return t

    # 1. 收尾句前统一成句号：匹配 [；;，,、] + 可选空格 + 收尾句
    t = re.sub(r'[；;，,、]\s*' + re.escape(_CLOSING), '。' + _CLOSING, t)
    # 收尾句前如果是句号但带了空格，也收干净
    t = re.sub(r'。\s*' + re.escape(_CLOSING), '。' + _CLOSING, t)

    # 清掉没配对的括号——模型偶尔抄了左括号却漏掉内容，
    # 生成「…自动采集（。期待…」这种残句
    t = re.sub(r'[（(]\s*[。；;，,、]', lambda m: m.group(0)[-1], t)
    t = re.sub(r'[（(]\s*$', '', t)
    t = re.sub(r'[（(][^）)]{0,2}[）)]', '', t)   # 空括号或只有 1-2 字的括号

    # 清掉本人明确表示不写的「异常兜底」类表述。
    # 提示词里已经写了禁令，但模型仍会偶发（实测 38 条里漏了 1 条），
    # 这类内容属于「没让写就别写」，用代码兜底比反复调提示词可靠。
    _EXC = r'(弹窗|异常|报错|超时|重试|崩了|故障)'
    # ① 「…弹窗之类的异常也能兜住」整段去掉
    t = re.sub(r'[，,；;]?\s*' + _EXC + r'[^，。；]{0,8}'
               r'(?:也能(?:自己)?(?:处理掉|兜住|兜底|处理|应付过去|扛住)|'
               r'也能自动处理|会自动处理|都能自己处理)[^，。；]{0,4}', '', t)
    # ② 「能自动处理弹窗这类意外情况」整段去掉
    t = re.sub(r'[，,；;]?\s*(?:能|可以|会)?自动处理' + _EXC + r'[^，。；]{0,8}', '', t)
    # ③ 「超时能自动重试」这类短语去掉
    t = re.sub(r'[，,；;]?\s*' + _EXC + r'\s*(?:时|后)?\s*(?:能|可以|会|自动)?\s*'
               r'(?:自动)?\s*(?:重试|恢复|兜底|处理|捕获)[^，。；]{0,4}', '', t)
    t = re.sub(r'。{2,}', '。', t)
    t = re.sub(r'[，,；;]\s*。', '。', t)     # 删完留下空逗号
    t = re.sub(r'。\s*[，,；;]', '。', t)
    # 删完只剩个孤立动词的，比如「…程序，遇到，不用人盯着」
    t = re.sub(r'(遇到|碰到|出现|发生|处理|支持)[，,]\s*', '', t)
    t = re.sub(r'[，,]\s*[，,]', '，', t)

    if t[-1] in '。！？':
        return t
    # 去掉悬空的标点
    core = t.rstrip('，,、；;：: ')
    if not core:
        return t
    if core[-1] in '。！？':
        return core
    if not core.endswith(_CLOSING.rstrip('。')) and len(core) + len(_CLOSING) <= limit:
        return core + '。' + _CLOSING
    return core + '。'


def _finalize(text, row, opening_unused=None):
    """统一收口。顺序很重要：

    1. 先补收尾句 / 去掉悬空标点（模型偶尔输出没写完的话）
    2. 再决定要不要附仓库链接
    3. 最后做长度兜底（反过来的话，附加的链接会把总长度顶超上限）
    """
    t = _ensure_closing(str(text or '').strip())
    return _cap_length(_attach_oss(t, row))


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

# ============================================================
# 以下代码已停用，保留供参考 —— 话术生成已从「按项目匹配」
# 改为「按能力清单匹配」（见 _CAPABILITIES / pick_capabilities / cap_say）。
# 此处没有调用点，保留是为了避免以后重新推导这段设计。
# ============================================================
# # ---------- 主打项目：同样由 Python 决定，不交给模型 ----------
# # 每个项目配一组「JD 命中词」。命中最多者优先；都不命中时按顺序轮换，
# # 避免 40 条话术全都讲同一个项目（实测同一批 10 条全讲金融终端，很像群发）。
# # 每个项目配三样东西：
# #   name  项目内部名（也用于降级模板）
# #   tools 可被念出来的技术名 —— 招呼语要**先亮这些词**，HR 扫一眼就能命中关键词
# #   desc  做出来能干嘛（业务价值），不要写成「我怎么实现的」
# #   keys  命中词，用于判断这个岗位该讲哪个项目
# #
# # 命名注意：不要把项目叫成「XX论文系统」——「论文」两字会让人一眼判定是课程作业。
# _PROJECTS = {
#     'ai': [
#         {
#             'name': '智能求职助手',
#             'tools': ['Chrome DevTools Protocol', 'TF-IDF', '大模型 API'],
#             'desc': '做过一个从数据采集到内容生成的全链路 AI 应用：自动采集结构化数据、'
#                     '用 TF-IDF 与余弦相似度做匹配排序、再调大模型 API 生成定制内容，'
#                     '并用网页面板把整条流程承接起来',
#             'keys': ['agent', '智能体', 'mcp', '工具调用', 'workflow', '编排',
#                      'function calling', '全栈', '端到端', '闭环'],
#         },
#         {
#             'name': '知识库问答',
#             'tools': ['LangChain', 'Chroma', 'BGE Embedding'],
#             'desc': '用这套做过一套面向企业内部文档的问答服务：把 PDF 等非结构化文档'
#                     '解析、切分、向量化后做语义检索，回答带原文出处，能直接接进业务系统',
#             'keys': ['rag', 'langchain', 'chroma', '向量', '知识库', '语义检索',
#                      'embedding', '召回', '检索增强', '问答'],
#         },
#         {
#             'name': 'AI图像批量生成',
#             'tools': ['Prompt Engineering', '结构化模板'],
#             'desc': '用结构化 Prompt 模板控制光源、色调、景别、场景等维度做批量生成，'
#                     '把出图一致性从六成提到九成以上，并沉淀成可复用的检查清单',
#             'keys': ['prompt', '提示词', 'aigc', '图像', '生图', '风格', '多模态', '文生图'],
#         },
#     ],
#     'qa': [
#         {
#             'name': '金融终端自动化导出系统',
#             'tools': ['Python', '图像识别定位', '模拟键鼠操作'],
#             'desc': '做过一套无人值守的桌面自动化程序：自动完成导出、加时间戳归档、'
#                     '按日期合并汇总并邮件发送，'
#                     '打包成 exe 让非技术同事直接双击使用，上线后每天省下 30 分钟人工操作',
#             'keys': ['自动化脚本', '定时任务', '无人值守', 'windows', '桌面',
#                      'exe', '批量处理', '运维'],
#         },
#         {
#             'name': '智能求职助手',
#             'tools': ['Chrome DevTools Protocol', '元素定位', '结构化采集'],
#             'desc': '用 CDP 控制浏览器采集结构化数据，针对元素失效、'
#                     '页面加载超时做了等待与异常处理，保证长时间运行不中断',
#             'keys': ['selenium', 'playwright', 'appium', 'ui自动化', '元素定位',
#                      '爬虫', '抓取', 'cdp', 'chromedriver', 'web自动化'],
#         },
#         {
#             'name': '接口联调与数据处理',
#             'tools': ['Python 接口调用', 'HTTP 协议', 'JSON 结构校验'],
#             'desc': '日常用 Python 做接口联调与数据清洗，能根据返回结构写字段校验与'
#                     '异常分支处理，把接口数据整理成可直接分析的表格',
#             # 只放「接口测试」这类专有说法。
#             # 早期误放了 'http'、'requests' 等泛词，导致 50% 测试岗都被判成这一项。
#             'keys': ['接口测试', '接口自动化', 'api测试', 'api自动化',
#                      'postman', 'jmeter', '接口联调'],
#         },
#         {
#             'name': '知识库问答',
#             'tools': ['LangChain', 'Chroma', 'RAG 链路验证'],
#             'desc': '搭过检索增强问答的完整链路，并对切分粒度、召回效果逐环节做过对比验证，'
#                     '能为 AI 类产品的效果评估提供可复用的方法',
#             # 这里只保留 AI 测试专用词。原先放了 '大模型''ai应用' 等泛词，
#             # 几乎每个 AI 岗都会命中，把本该分给其它项目的岗位全抢走了。
#             'keys': ['ai测试', '模型评测', '大模型测试', 'rag', 'llm'],
#         },
#     ],
# }
# _project_cursor = {}


# ============================================================
# 能力清单：按「岗位要求 → 能力 → 一句话说法」组织
# ============================================================
# 生成话术时不再「先定讲哪段经历」，而是把整份清单交给模型，
# 让它按 JD 挑 2-3 条最对得上的能力来讲。表述组合本身在变，
# 就不会再出现「同一段经历换 72 种说法」。
#
# level 的含义：
#   'strong' —— 技术上硬、可验证，优先挑
#   'weak'   —— 比较基础（本人自评），除非岗位明确要求，否则不挑
# keys 用于和 JD 匹配，命中的能力在提示词里会被标注出来
# 每条能力的 say 字段写法要求（踩过坑，改前先读）：
#   先说「做出一个什么东西、它能干什么」，技术名放句中当佐证。
#
#   反例（早期版本）：我做过带大模型API的问答工具，调用失败能退回备用文案
#     → 这是实现说明，HR 不知道这是什么，"API 挂了怎么办"更不是他关心的事
#   正例：做过文档问答系统，上传 PDF 就能提问，答案带原文出处方便核对
#     → 对方一眼知道这东西干什么用
_CAPABILITIES = [
    # ---------- 硬实力 ----------
    # 说明：这一条是本轮新增的。原先能力清单里没有「智能体/工作流」这一类，
    # 导致投「AI 智能体应用工程师」这类岗位时无能力可选，
    # 只能退而挑「大模型 API 集成」「自动化」，跟岗位主线对不上
    # （实测精创仪器那份 JD 主线就是智能体 + 工作流，话术却只讲了大模型和采集）。
    {'level': 'strong', 'name': '智能体与工作流搭建',
     'says': [
         '把大模型和业务系统串起来做成自动化流程，多个步骤能自己往下走，不用人一步步操作',
         '搭过带工具调用的智能体：让它自己取数、判断、再调用接口完成后续动作',
         '设计过端到端的工作流，把采集、处理、生成、输出几步串成一条自动链路',
     ],
     'keys': ['agent', '智能体', 'mcp', '工作流', 'workflow', '编排', '工具调用',
              'function calling', '多轮', 'coze', 'dify', 'langgraph', 'n8n', 'agent开发']},
    {'level': 'strong', 'name': '大模型 API 集成',
     'says': [
         '把大模型接进业务流程，让它自动生成和整理内容，替掉原来靠人重复写的部分',
         '做的系统会调用大模型来处理内容，不用人工一句句去写',
         '把大模型能力接进现有流程里跑通，让机器承担重复的文字工作',
     ],
     'keys': ['大模型', 'llm', 'api', 'deepseek', 'chatglm', '模型调用', 'gpt']},
    {'level': 'strong', 'name': '浏览器自动化采集',
     'says': [
         '写过自动采集数据的程序，替代人工复制粘贴，结果直接落成规范表格',
         '让程序自己去页面上取数，不用人一条条抄，采完就是整理好的表格',
         '做过网页数据自动抓取，把原来手工搬数据的活交给程序',
     ],
     'keys': ['cdp', '爬虫', '采集', '抓取', 'selenium', 'playwright']},
    {'level': 'strong', 'name': '匹配排序算法',
     'says': [
         '做过相关性排序，把杂乱信息按匹配度排出优先级，省掉人工逐条筛的时间',
         '写过一套打分排序逻辑，自动把最相关的排在前面，不用一条条看',
         '用相似度算法给信息自动排序，把人工筛选的活减掉大半',
     ],
     'keys': ['tf-idf', '相似度', '排序', '匹配算法', '召回', '检索排序']},
    {'level': 'strong', 'name': '无人值守自动化',
     'says': [
         '做过定时自动运行的程序，导出、归档、发邮件全程不用人盯着，每天省下半小时',
         '把每天重复的手工操作写成定时任务，跑完还自动发邮件通知，人不用守着',
         '写过无人值守的自动流程，到点自己跑，中间不需要人干预',
     ],
     'keys': ['自动化', '定时', '无人值守', 'rpa', '批量处理', '运维']},
    {'level': 'strong', 'name': 'Excel 数据处理',
     'says': [
         '把导出的表格自动加时间戳归档、按日期合并汇总，解决文件互相覆盖的问题',
         '做过表格自动归档与合并，历史数据能按日期追溯，不会再被新文件覆盖',
         '用程序替代手工整理表格，自动打时间戳、按天汇总，省掉反复复制粘贴',
     ],
     'keys': ['excel', 'openpyxl', '表格', '报表', '数据处理']},
    # ---------- 基础能力：岗位明确要求时才讲 ----------
    {'level': 'weak', 'name': '文档解析与切分',
     'says': [
         '把 PDF 这类不好检索的资料解析、按语义切分后存起来，后面就能直接搜内容',
         '做过文档预处理，把零散的 PDF 整理成可检索的结构化数据',
     ],
     'keys': ['pdf', '文档', '解析', '切分', '非结构化']},
    {'level': 'weak', 'name': '向量检索',
     'says': [
         '把文档转成向量做语义检索，用大白话提问也能找到对应内容，比关键词匹配找得全',
         '做过语义检索，用户不用记准确的关键词，按意思问就能搜到',
     ],
     'keys': ['向量', 'chroma', 'embedding', '语义检索', '召回', 'milvus', 'faiss']},
    {'level': 'weak', 'name': 'RAG 问答系统',
     'says': [
         '做过文档问答系统，上传 PDF 就能提问，答案带原文出处方便核对',
         '把公司资料做成可提问的知识库，问什么答什么，还能点回原文验证',
         '搭过一套问答系统，把 PDF 等资料变成能直接提问的知识入口',
         '做过资料问答工具：把散落的文档收进一个库，用自然语言问就能定位到内容',
         '把一堆 PDF 整理成可直接检索的知识库，提问后答案会标出出自哪一句',
         '做过基于文档的问答，用户不用翻资料，问一句就能拿到答案和出处',
     ],
     'keys': ['rag', 'langchain', '检索增强', '知识库']},
    {'level': 'weak', 'name': '接口调用与校验',
     'says': [
         '写过接口联调脚本，把多个系统的数据对接起来，返回内容会做校验',
         '做过系统间数据对接，接口返回不对能第一时间发现，而不是等业务出错',
     ],
     'keys': ['接口', 'api测试', 'requests', 'http', 'postman']},
    {'level': 'weak', 'name': 'Prompt 工程',
     'says': [
         '用结构化提示词模板控制 AI 输出，同样的需求每次结果保持一致',
         '把提示词写成固定模板，让 AI 产出稳定可复现，而不是每次都不一样',
     ],
     'keys': ['prompt', '提示词', 'aigc', '文生图', '生图', '多模态']},
    {'level': 'weak', 'name': '数据清洗与分析',
     'says': [
         '用 Python 做数据清洗与统计汇总，把原始数据整理成能直接看的结论',
         '做过数据整理，把杂乱原始数据清洗后汇总成可直接使用的报表',
     ],
     'keys': ['pandas', '数据清洗', '数据分析', 'numpy']},
]


# 每条能力的说法轮换游标。
# 同一批里讲到同一条能力的次数可能很多次，如果每次都用同一句素材，
# 生成的话术前半句会几乎一样（实测第 4、5、11 条完全雷同），
# 所以按序轮换 says 里的不同说法。
_cap_cursor = {}


def cap_say(cap):
    """取这条能力的下一个说法（轮换）。"""
    says = cap.get('says') or [cap.get('say', '')]
    i = _cap_cursor.get(cap['name'], 0)
    _cap_cursor[cap['name']] = i + 1
    return says[i % len(says)]


def pick_capabilities(row, limit=3):
    """按 JD + 标题挑能力，返回 (选中列表, 命中的关键词集合)。

    排序规则（按优先级从高到低）：
      1. strong 且命中标题   —— 岗位主线就是它，最该讲
      2. strong 且命中 JD    —— 硬实力且对口
      3. weak 且命中 JD      —— 岗位明确要求，可以讲
      4. strong 未命中       —— 硬实力但不对口，兜底用
      5. weak 未命中         —— 不讲

    两个踩过的坑：
      · 不能把 strong 当权重加在命中数上——那样 weak 命中 3 个词会压过
        strong 命中 2 个词，硬实力被挤掉（实测 AI 岗选出来全是 weak）。
      · 「命中标题」必须单独排在「命中 JD」前面。JD 里往往罗列一堆加分项，
        而标题才是岗位主线。实测：投「AI智能体应用工程师」时，JD 里既有
        "智能体"也有"自动化""大模型"，不区分标题就会挑到后面几个，
        话术讲的和岗位主线对不上。
    """
    jd = str(row.get('jd', '')).lower()
    title = str(row.get('title', '')).lower()

    strong_title, strong_body, weak_body, strong_miss = [], [], [], []
    for cap in _CAPABILITIES:
        t_hits = [k for k in cap['keys'] if k in title]
        j_hits = [k for k in cap['keys'] if k in jd]
        if t_hits:
            (strong_title if cap['level'] == 'strong' else weak_body).append(
                (len(t_hits) * 10 + len(j_hits), cap, t_hits + j_hits))
        elif j_hits:
            (strong_body if cap['level'] == 'strong' else weak_body).append(
                (len(j_hits), cap, j_hits))
        elif cap['level'] == 'strong':
            strong_miss.append((0, cap, []))

    for b in (strong_title, strong_body, weak_body, strong_miss):
        b.sort(key=lambda x: -x[0])

    ordered = strong_title + strong_body + weak_body + strong_miss
    chosen = ordered[:limit]
    return [c[1] for c in chosen], {k for c in chosen for k in c[2]}

# ============================================================
# 以下代码已停用，保留供参考 —— 话术生成已从「按项目匹配」
# 改为「按能力清单匹配」（见 _CAPABILITIES / pick_capabilities / cap_say）。
# 此处没有调用点，保留是为了避免以后重新推导这段设计。
# ============================================================
# # 关键词全都没命中时，默认讲哪个项目。
# # QA 用金融终端（测试岗硬通货：无人值守 + 异常处理 + 打包交付）；
# # AI 用智能求职助手（技术链路最全：采集 + 匹配算法 + 大模型 API + 前端面板）。
# DEFAULT_PROJECT = {
#     'ai': '智能求职助手',
#     'qa': '金融终端自动化导出系统',
# }
#
#
# def _pick_project(row, profile):
#     """按 JD 关键词选主打项目；都不命中时用默认项目，并按顺序轮换保证分布。"""
#     global _project_cursor
#     jd = str(row.get('jd', '')).lower()   # 只看 JD，标题里带自己项目名会造成自匹配
#     specs = _PROJECTS.get(profile['track'], [])
#     if not specs:
#         return None, 0
#
#     scored = []
#     for i, p in enumerate(specs):
#         hits = sum(1 for k in p['keys'] if k in jd)
#         scored.append((hits, i, p))
#
#     best_hits = max((s[0] for s in scored), default=0)
#     if best_hits > 0:
#         # 命中最多者优先；并列时轮换，避免永远只选第一个
#         tied = [s for s in scored if s[0] == best_hits]
#         cur = _project_cursor.get(profile['track'], 0)
#         chosen = tied[cur % len(tied)]
#         _project_cursor[profile['track']] = cur + 1
#         return chosen[2], best_hits
#
#     # 无命中：用默认项目（找得到就用，找不到退回第一个）
#     default_name = DEFAULT_PROJECT.get(profile['track'], specs[0]['name'])
#     for p in specs:
#         if p['name'] == default_name:
#             return p, 0
#     return specs[0], 0


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

    # 候选能力分两组交给模型。分组展示 + 硬规则，避免模型凭「感觉最贴」自己挑，
    # 实测：只给一个混合列表 + 「优先挑★」这种软要求时，模型会跳过 strong 去选 weak
    # （40 条里 39 条都讲了 RAG，尽管 strong 排在列表前面）。
    caps, cap_hits = pick_capabilities(row, limit=5)
    strong_caps = [c for c in caps if c['level'] == 'strong']
    weak_caps = [c for c in caps if c['level'] == 'weak']

    def _fmt(c):
        matched = [k for k in c['keys'] if k in (jd + title).lower()]
        # 只有岗位 JD 真的提到某项英文技术时，才把它作为「可提及的技术名」给出来；
        # JD 没提的就不给，避免模型在句尾硬贴一个对方用不上的名词。
        tech = [k for k in c['keys'] if k in jd.lower() and re.match(r'^[A-Za-z]', k)]
        line = f"   · {c['name']}：{cap_say(c)}"
        if tech:
            line += f"（岗位提到 {('、'.join(tech[:2]))}，可以顺带提一下）"
        return line

    strong_txt = '\n'.join(_fmt(c) for c in strong_caps) or '   （无）'
    weak_txt = '\n'.join(_fmt(c) for c in weak_caps) or '   （无）'

    # JD 技术词只作为「这个岗位在意什么」的提示，且只允许在硬实力缺席时才起作用。
    # 早期写成「岗位提到 X，优先讲那条」，结果与「一律只用硬实力」直接冲突，
    # 模型反而被引导去讲 RAG（因为 JD 里 RAG 词最多），等于自己拆自己的台。
    jd_tech = extract_jd_tech(jd)
    if jd_tech and not strong_caps:
        tech_rule = (
            f"3. 这个岗位明确提到了 {'、'.join(jd_tech)}，"
            f"从上面的基础能力里挑最贴的一条讲。\n"
        )
    else:
        tech_rule = ""

    # 画像摘要只截前 200 字：完整摘要里塞了 4 个项目，模型会贪心全写进去
    profile_brief = profile['personal_summary'].strip()[:200]

    prompt = f"""给下面这个岗位写一段打招呼语。

【求职者】
{profile_brief}

【目标岗位】
公司：{company}
职位：{title}
岗位描述：{jd_excerpt}

【硬实力（必须从这里挑，讲了就是加分项）】
{strong_txt}

【基础能力（只在硬实力完全对不上、而岗位又明确提及时才可用）】
{weak_txt}

【硬性要求】
1. 开头必须原样照抄这一句，不要改写：
   「{opening}」
   除非 JD 明确写了"欢迎应届生/校招"，否则不要出现"2026届""应届生""毕业生"。
2. **从【硬实力】里挑 1-2 条来讲。**
   规则：
   - 【硬实力】里有内容时，**一律只用硬实力，不许改用基础能力**
   - 只有当【硬实力】是（无），或者确实一条都对不上时，才从基础能力里挑
   - **先说要做出一个什么东西、它能干什么，再顺带提技术名。**
     不要反过来先报技术名、也不要解释技术是怎么实现的。
     反例：「我做过带大模型API的问答工具，调用失败能退回备用文案」
           ——对方不知道这是什么，"API 挂了怎么办"更不是他关心的事
     正例：「做过文档问答系统，上传 PDF 就能提问，答案带原文出处方便核对」
           ——一眼就知道这东西干什么用
   - 不要复述整份清单，也不要罗列技术名词
   - 不要报项目名称，说事就行
   - **用你自己的话改写，不要照抄候选里的句子**——下面只是素材，
     照着抄的话几十条话术会一模一样。结合这个岗位的实际业务来说。
     改写方式举例：换主语（我做过 / 手里做过一个 / 之前落地过）、
     换动词（能提问 / 直接问就行 / 拿自然语言就能查到）、
     换落点（方便核对 / 省掉翻资料的时间 / 不用再逐条找）。
   - 标注了「岗位提到 XX」的，可以顺带提一下那个技术名；
     没标注的就不要硬塞技术名，对方用不上。
{tech_rule}
【风格】
- 像人在跟人说话，不要写成简历摘要
- 禁用这类空话：「闭环」「赋能」「全方位」「高效」「差异化」「全流程」
- 不要写"面试回复率提升XX%"这类个人求职数据，也不要放网址或提 GitHub
- 结尾用"期待有机会进一步沟通"，**前面要用句号，不要用分号或逗号连接**

【不许编造——这条最重要】
上面能力清单里怎么写的，你就怎么说，**不要替它加细节**。
举个真实的反例：清单写的是「调用失败能退回备用文案」，
模型自己发挥成了「主模型挂了也能自动切换」——**这是编的，不许这样写**。
不确定的技术名词、没提到的机制，一律不要写进话术。

具体禁止补充这几类细节（都是模型自己加过的）：
- **不要提「弹窗」「异常兜底」「超时重试」**——求职者明确表示这些不写
- 不要写「能出差」这类未经提供的个人承诺
- 不要留下孤立的括号，比如「…自动采集（。」

【输出】
直接给招呼语正文，不要引号、不要解释。总长 100 字以内。

招呼语："""

    system_prompt = (
        "你是一位有 5 年经验的求职顾问，熟悉招聘方的阅读习惯：\n"
        "初筛 HR 平均 3 秒扫一条消息，他只看三件事——\n"
        "这个人能干这个活吗？他跟岗位哪里对得上？他靠谱吗？\n\n"
        "你的任务不是复述求职者的经历，而是从给到的能力清单里挑出最对得上的，"
        "用具体、自然的话讲出来。宁可讲一个真实的小细节，也不要讲一堆正确的空话。"
    )

    try:
        resp = requests.post(
            DEEPSEEK_CONFIG['base_url'],
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            json={
                'model': DEEPSEEK_CONFIG['model'],
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt},
                ],
                'temperature': 0.85,
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
GREETING_LIMIT = 60   # 每个赛道最多给多少个岗位生成招呼语（要调大就改这里）


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

    # 重置轮换游标，保证每次运行的起点一致（否则多次运行会从上次的位置接着轮）
    _cap_cursor.clear()
    _opening_seq.clear()

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

        # 硬排除：方向不符 / 核心技术栈不符 / 外包 / 5 年经验门槛
        reasons = sub.apply(lambda r: _exclude_reason(r, profile), axis=1)
        hit = reasons.str.len() > 0
        if hit.any():
            print(f"   已剔除 {int(hit.sum())} 个不符合的岗位：")
            shown = 0
            for idx, why in reasons[hit].items():
                if shown >= 10:
                    print(f"      … 另有 {int(hit.sum()) - shown} 个")
                    break
                print(f"      ✂ {str(sub.loc[idx, 'title'])[:34]:36s} "
                      f"| {str(sub.loc[idx, 'boss_name'])[:16]:16s} | {'、'.join(why)}")
                shown += 1
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
            g = generate_greeting(row, profile)
            # 最外层保险：不管上游哪条路径返回的，落盘前再过一次收尾处理。
            # generate_greeting 内部已经调过 _finalize，这里再兜一次是因为
            # 实测出现过 8/40 条话术仍带「；期待有机会进一步沟通。」的情况，
            # 而单点测试却无法复现——与其继续追根因，不如在唯一写入口加防线。
            sub.loc[idx, 'greeting'] = _finalize(g, row)
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
