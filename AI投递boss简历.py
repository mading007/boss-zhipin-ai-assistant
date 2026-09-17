import json
import re
import ast
import pandas as pd
import requests
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 第一步：填写你的【个人简历画像】
# ============================================================
MY_PROFILE = {
    'tech_skills': ['Python', 'SQL', 'Excel', 'Tableau', 'Power BI', 'Pandas', 'NumPy', 'Matplotlib', 'LangChain',
                    'Chroma', 'AI应用开发'],
    'years_exp': 0.0,
    'education': '本科',
    'expected_salary': 15,
    'preferred_company_size': 0,
    'personal_summary': """
我是马丁，2026届电子信息工程本科毕业生，可立即到岗。

核心项目：
1. BOSS直聘智能求职助手（9月）：独立开发爬虫采集400+岗位，设计TF-IDF匹配算法，接入大模型API生成差异化招呼语，形成15秒高效投递闭环，面试回复率提升至28%；
2. RAG论文问答系统（8月）：基于LangChain+Chroma开发，支持PDF上传、语义检索与答案溯源，代码已开源；
3. AI图像风格统一生成（6月）：设计结构化Prompt模板（光源/色调/景别/场景五维控制），将风格统一度从60%提升至90%以上；
4. 金融交易终端自动导出系统（6月）：通过图像识别+模拟键鼠操作，实现每日定时导出并自动发送邮件。

技术栈：Python、Pandas、SQL、LangChain、Chroma、Prompt Engineering、自动化脚本。
""",
    'greeting_style': 'professional',
    'your_name': '马丁'
}

# ============================================================
# 第二步：配置 DeepSeek API
#    密钥存放在同目录的 .env 文件里（不要写死在代码中，也不要把 .env 提交到 Git）
#        DEEPSEEK_API_KEY=sk-xxxx
#    不配置也能运行，会降级为模板版招呼语
# ============================================================
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_PATH, override=True)
except ImportError:
    if os.path.exists(_ENV_PATH):
        with open(_ENV_PATH, 'r', encoding='utf-8') as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith('#') or '=' not in _line:
                    continue
                _k, _, _v = _line.partition('=')
                os.environ[_k.strip()] = _v.strip().strip('"').strip("'")

DEEPSEEK_CONFIG = {
    'api_key': os.environ.get('DEEPSEEK_API_KEY', '').strip(),
    'model': os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat'),
    'base_url': 'https://api.deepseek.com/v1/chat/completions'
}
# ============================================================

# ---------- 自动查找最新的 BOSS 数据文件 ----------
boss_dir = r'C:\Users\madin\.boss-zhipin-scraper\job-result'
job_files = glob.glob(os.path.join(boss_dir, 'boss_jobs_*.json'))
job_files = [f for f in job_files if 'merged' not in f]

if not job_files:
    raise FileNotFoundError("❌ 未找到任何 boss_jobs_*.json 文件，请先运行爬虫")

latest_job = max(job_files, key=os.path.getctime)
print(f"📂 使用最新数据文件: {os.path.basename(latest_job)}")

with open(latest_job, 'r', encoding='utf-8') as f:
    raw = json.load(f)
jobs = raw['jobs']
df = pd.DataFrame(jobs)
print(f"📂 加载合并数据，共 {len(df)} 条岗位（BOSS + 猎聘）")


# ---------- 解析函数 ----------
def parse_experience(tags):
    if not isinstance(tags, str):
        return None
    parts = tags.split('|')
    for p in parts:
        if '年' in p:
            return p.strip()
    return None

def parse_education(tags):
    if not isinstance(tags, str):
        return None
    parts = tags.split('|')
    for p in parts:
        p = p.strip()
        if p in ['大专', '本科', '硕士', '博士', '学历不限']:
            return p
    return None

def parse_salary_mid(salary_str):
    if not isinstance(salary_str, str):
        return None
    match = re.search(r'(\d+\.?\d*)-(\d+\.?\d*)K', salary_str)
    if match:
        return (float(match.group(1)) + float(match.group(2))) / 2
    return None

def parse_tech_stack(row):
    tech = row.get('tech_stack', [])
    if isinstance(tech, str):
        try:
            return ast.literal_eval(tech)
        except:
            return []
    return tech if isinstance(tech, list) else []

df['tech_list'] = df.apply(parse_tech_stack, axis=1)


# ---------- 匹配打分 ----------
def calculate_match_score(row):
    score, details = 0, {}
    W_TECH, W_EXP, W_EDU, W_SCALE, W_SALARY, W_SEMANTIC = 30, 20, 10, 10, 10, 35

    jd_tech, my_tech = row['tech_list'], MY_PROFILE['tech_skills']
    if jd_tech and my_tech:
        matched = len(set(my_tech) & set(jd_tech))
        tech_score = (matched / len(jd_tech)) * W_TECH
    else:
        tech_score = W_TECH * 0.3
    score += tech_score
    details['技术栈'] = round(tech_score, 1)

    exp_req, my_exp = parse_experience(row.get('tags', '')), MY_PROFILE['years_exp']
    if exp_req:
        if '不限' in exp_req:
            exp_score = W_EXP * 0.8
        else:
            nums = re.findall(r'(\d+)', exp_req)
            if len(nums) >= 2 and int(nums[0]) <= my_exp <= int(nums[1]):
                exp_score = W_EXP
            elif my_exp < int(nums[0]):
                exp_score = W_EXP * 0.5 if my_exp >= 0.5 else W_EXP * 0.3
            else:
                exp_score = W_EXP * 0.8
    else:
        exp_score = W_EXP * 0.6
    score += exp_score
    details['经验'] = round(exp_score, 1)

    edu_req = parse_education(row.get('tags', ''))
    edu_rank = {'大专': 1, '本科': 2, '硕士': 3, '博士': 4}
    if edu_req:
        if edu_req == '学历不限':
            edu_score = W_EDU
        else:
            edu_score = W_EDU if edu_rank.get(MY_PROFILE['education'], 0) >= edu_rank.get(edu_req, 0) else W_EDU * 0.2
    else:
        edu_score = W_EDU * 0.7
    score += edu_score
    details['学历'] = round(edu_score, 1)

    pref_size = MY_PROFILE['preferred_company_size']
    scale = row.get('company_scale', '')
    if pref_size > 0:
        if '10000' in scale:
            scale_score = W_SCALE
        elif '1000' in scale:
            scale_score = W_SCALE * 0.8 if pref_size >= 1000 else W_SCALE * 0.5
        else:
            scale_score = W_SCALE * 0.3
    else:
        scale_score = W_SCALE * 0.5
    score += scale_score
    details['公司规模'] = round(scale_score, 1)

    sal_mid = parse_salary_mid(row.get('salary', ''))
    if sal_mid and MY_PROFILE['expected_salary'] > 0:
        ratio = min(sal_mid, MY_PROFILE['expected_salary']) / max(sal_mid, MY_PROFILE['expected_salary'])
        salary_score = W_SALARY if ratio >= 0.9 else W_SALARY * 0.5 if ratio >= 0.7 else W_SALARY * 0.2
    else:
        salary_score = W_SALARY * 0.3
    score += salary_score
    details['薪资匹配'] = round(salary_score, 1)

    jd_text, my_text = row.get('jd', ''), MY_PROFILE.get('personal_summary', '')
    if jd_text and my_text and len(jd_text) > 50:
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            tfidf = vectorizer.fit_transform([my_text, jd_text])
            sem_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * W_SEMANTIC
        except:
            sem_score = W_SEMANTIC * 0.3
    else:
        sem_score = W_SEMANTIC * 0.3
    score += sem_score
    details['语义匹配'] = round(sem_score, 1)

    row['detail_scores'] = str(details)
    return round(score, 2)

df['match_score'] = df.apply(calculate_match_score, axis=1)
df_sorted = df.sort_values('match_score', ascending=False)

# ---------- 过滤硕士岗位 ----------
df_sorted = df_sorted[~df_sorted['tags'].str.contains('硕士', na=False)]
print(f"🔍 已剔除硕士岗，剩余 {len(df_sorted)} 条可投递")

# ---------- 生成地区列 ----------
df_sorted['district'] = df_sorted['location'].str.split('·').str[1]


# ---------- 招呼语生成（针对 AI 应用岗位优化） ----------
def generate_greeting_with_ai(row):
    if not DEEPSEEK_CONFIG['api_key'] or DEEPSEEK_CONFIG['api_key'] == 'sk-你的DeepSeek API密钥':
        return _fallback_greeting(row)

    title = row.get('title', '该岗位')
    jd_text = row.get('jd', '')
    if len(jd_text) > 800:
        jd_text = jd_text[:800] + '...'

    tech_str = ', '.join(row['tech_list'][:5]) if row['tech_list'] else '数据分析相关技术'
    my_summary = MY_PROFILE.get('personal_summary', '').strip()
    style = MY_PROFILE.get('greeting_style', 'professional')
    style_map = {
        'professional': '正式、专业、诚恳',
        'casual': '轻松、友好、有人情味',
        'enthusiastic': '热情、积极主动、有冲劲'
    }
    style_desc = style_map.get(style, '正式、专业')

    prompt = f"""请根据以下信息，为求职者撰写一段发给HR的打招呼语（60-90字）：

    【求职者背景】
    {my_summary}
    掌握技术：{tech_str}
    学历：本科

    【目标岗位】
    职位：{title}
    岗位描述：{jd_text}

    【要求】
    1. 称呼统一用"您好"，不要用"尊敬的XXX"。
    2. 第一句直接说："我是{MY_PROFILE['your_name']}。" 不要提"2026届""应届生""本科毕业生"等身份标签。
    3. 第二句：**默认写"智能求职助手"项目**（爬虫 + TF-IDF匹配 + 大模型API + HTML面板，面试回复率提升至28%），突出Python、爬虫、大模型API调用能力。
       例外情况（仅当满足以下条件时才切换项目）：
       - 如果JD中明确出现"RAG""LangChain""Chroma""向量数据库""语义检索""Embedding"等关键词 → 改写"基于LangChain+Chroma的RAG论文问答系统"
       - 如果JD中明确出现"Prompt Engineering""图像生成""AIGC图像"等关键词 → 改写"AI图像风格统一生成"
    4. 第三句：表达"能快速上手AI应用开发"的意思，用"期待有机会进一步沟通"结尾。
    5. 不要写"面试回复率提升至XX%"这种个人求职数据。
    6. 不要出现"GitHub""开源""代码可查"等词。
    7. 如果JD里明确写了"欢迎应届生"或"校招"，可以提及"应届生"身份；否则任何地方都不要出现"应届""2026届""毕业生"等词。
    8. 语气：正式、专业、诚恳。
    9. 直接输出招呼语正文，不要加任何额外说明。

    招呼语："""
    try:
        headers = {
            'Authorization': f'Bearer {DEEPSEEK_CONFIG["api_key"]}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': DEEPSEEK_CONFIG['model'],
            'messages': [
                {'role': 'system', 'content': '你是一位专业的求职顾问，擅长帮求职者写出真诚、有吸引力、突出项目能力的打招呼语。'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.8,
            'max_tokens': 250
        }

        print(f"⏳ 请求: {row.get('title', '')[:20]}...", end='', flush=True)
        response = requests.post(
            DEEPSEEK_CONFIG['base_url'],
            headers=headers,
            json=payload,
            timeout=5
        )
        print(" 完成")
        if response.status_code == 200:
            result = response.json()
            greeting = result['choices'][0]['message']['content'].strip()
            greeting = greeting.strip('"').strip()
            return greeting
        else:
            print(f" 状态码 {response.status_code}，使用模板")
            return _fallback_greeting(row)
    except requests.exceptions.Timeout:
        print(" 超时，使用模板")
        return _fallback_greeting(row)
    except Exception as e:
        print(f" 异常: {e}，使用模板")
        return _fallback_greeting(row)

def _fallback_greeting(row):
    title = row.get('title', '该岗位')
    name = MY_PROFILE['your_name']
    tech_list = row['tech_list']
    my_tech = MY_PROFILE['tech_skills']
    matched_tech = [t for t in tech_list if t in my_tech]
    tech_str = '、'.join(matched_tech[:3]) if matched_tech else '数据分析技能'
    return f"您好，我是{name}，对{title}岗位很感兴趣。我熟悉{tech_str}，有丰富的项目实践经历，希望能有机会进一步交流。"

print("\n🤖 正在生成招呼语（如未配置API则使用模板）...")
df_sorted['greeting'] = df_sorted.apply(generate_greeting_with_ai, axis=1)

# ---------- 输出结果 ----------
print("\n" + "=" * 80)
print("📋 AI 投递清单 TOP 10（含招呼语预览）")
print("=" * 80 + "\n")

for i, (idx, row) in enumerate(df_sorted.head(10).iterrows(), 1):
    print(f"{i:2d}. 匹配分: {row['match_score']:.1f} | {row.get('boss_name', '未知公司')[:25]} | {row['title'][:20]}")
    print(f"   薪资: {row['salary']} | 地区: {row.get('district', '未标注')}")
    print(f"   链接: {row.get('job_link', '')}")
    print(f"   💬 {row['greeting']}")
    print("-" * 70)

# ---------- 导出 ----------
output_cols = ['boss_name', 'title', 'salary', 'district', 'company_scale',
               'job_link', 'match_score', 'greeting']
df_sorted[output_cols].to_csv('apply_ready_list_ai.csv', index=False, encoding='utf-8-sig')

# ---------- HTML面板（使用 job_id 作为唯一标识） ----------
html_content = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI 投递助手</title>
<style>
    body { font-family: 'Segoe UI', sans-serif; background: #f5f7fa; padding: 20px; }
    .header-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; flex-wrap: wrap; }
    .btn-group button { margin-left: 10px; padding: 6px 15px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
    .btn-applied { background: #17a2b8; color: white; }
    .btn-ignored { background: #ffc107; color: #333; }
    .btn-reset { background: #dc3545; color: white; }
    table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #e0e4e8; }
    th { background: #2c3e50; color: white; font-weight: 600; }
    tr:hover { background: #f1f5f9; }
    .match-score { font-weight: bold; color: #2c3e50; }
    .btn-copy { background: #3498db; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; margin-right: 3px; }
    .btn-copy:hover { background: #2980b9; }
    .btn-done { background: #28a745; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; margin-right: 3px; }
    .btn-done:hover { background: #218838; }
    .btn-ignore { background: #ff9800; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; }
    .btn-ignore:hover { background: #e68900; }
    .toast { position: fixed; bottom: 20px; right: 20px; background: #2ecc71; color: white; padding: 10px 20px; border-radius: 6px; display: none; z-index: 999; }
    .greeting-preview { font-size: 13px; color: #555; max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: inline-block; }
    .btn-done.marked { background: #6c757d; cursor: not-allowed; }
    .row-done td { background: #f0f0f0; color: #999; }
    .row-ignored td { background: #fff3cd; color: #856404; }

    .modal-overlay {
        display: none;
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0,0,0,0.5);
        z-index: 1000;
        justify-content: center;
        align-items: center;
    }
    .modal-overlay.active { display: flex; }
    .modal-box {
        background: white;
        border-radius: 8px;
        max-width: 800px;
        width: 90%;
        max-height: 80vh;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        overflow: auto;
        position: relative;
    }
    .modal-box h3 { margin-top: 0; }
    .modal-box table { font-size: 14px; }
    .modal-box th { background: #e9ecef; color: #333; }
    .modal-close {
        position: sticky;
        top: 0;
        float: right;
        background: #dc3545;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 5px 15px;
        cursor: pointer;
        font-size: 16px;
    }
    .modal-close:hover { background: #c82333; }
</style>
</head>
<body>
<div class="header-bar">
    <h2>📋 AI 匹配投递清单</h2>
    <div class="btn-group">
        <button id="btnShowApplied" class="btn-applied">📋 已投递</button>
        <button id="btnShowIgnored" class="btn-ignored">📋 已忽略</button>
        <button id="btnResetApplied" class="btn-reset">🔄 重置</button>
    </div>
</div>
<p>按匹配分从高到低排序，点击 <strong>"📋 一键投递"</strong> 自动复制招呼语 + 打开链接 + 标记已投递。</p>
<p><small>💡 如果暂时不想投递，可点击 "📋 仅复制" 单独复制招呼语；"👎 不感兴趣" 可隐藏该岗位。</small></p>
<table id="jobTable">
<thead>
<tr>
    <th>匹配分</th>
    <th>公司</th>
    <th>职位</th>
    <th>薪资</th>
    <th>地区</th>
    <th>操作</th>
</tr>
</thead>
<tbody>
"""

for _, row in df_sorted.head(50).iterrows():
    boss_name = row.get('boss_name', '未知公司')
    title = row.get('title', '未知职位')
    salary = row.get('salary', '面议')
    district = row.get('district', '未知')
    score = row['match_score']
    link = row.get('job_link', '#')
    greeting = row['greeting'].replace('"', '&quot;').replace('\n', ' ')
    preview = greeting[:30] + '...' if len(greeting) > 30 else greeting
    job_id = row.get('job_id', link)

    html_content += f"""
    <tr data-job-id="{job_id}" data-company="{boss_name}" data-title="{title}" data-salary="{salary}" data-district="{district}" data-link="{link}">
        <td class="match-score">{score:.1f}</td>
        <td>{boss_name}</td>
        <td>{title}</td>
        <td>{salary}</td>
        <td>{district}</td>
        <td>
            <button class="btn-copy" data-greeting="{greeting}">📋 仅复制</button>
            <button class="btn-done" data-job-id="{job_id}" data-link="{link}" data-greeting="{greeting}">📋 一键投递</button>
            <button class="btn-ignore" data-job-id="{job_id}">👎 不感兴趣</button>
            <span class="greeting-preview" title="{greeting}">{preview}</span>
        </td>
    </tr>
    """

html_content += """
</tbody></table>
<div id="toast" class="toast">✅ 招呼语已复制，正在打开链接...</div>

<!-- 模态框：已投递 -->
<div id="appliedModal" class="modal-overlay">
    <div class="modal-box">
        <button class="modal-close" id="modalCloseApplied">✕ 关闭</button>
        <h3>📋 已投递岗位列表</h3>
        <div id="appliedListContent"><p>加载中...</p></div>
    </div>
</div>

<!-- 模态框：已忽略 -->
<div id="ignoredModal" class="modal-overlay">
    <div class="modal-box">
        <button class="modal-close" id="modalCloseIgnored">✕ 关闭</button>
        <h3>👎 已忽略岗位列表</h3>
        <div id="ignoredListContent"><p>加载中...</p></div>
    </div>
</div>

<script>
(function() {
    var STORAGE_KEY_APPLIED = 'appliedJobs';
    var STORAGE_KEY_IGNORED = 'ignoredJobs';

    function getApplied() {
        try { return JSON.parse(localStorage.getItem(STORAGE_KEY_APPLIED) || '[]'); } catch(e) { return []; }
    }
    function setApplied(list) { localStorage.setItem(STORAGE_KEY_APPLIED, JSON.stringify(list)); }

    function getIgnored() {
        try { return JSON.parse(localStorage.getItem(STORAGE_KEY_IGNORED) || '[]'); } catch(e) { return []; }
    }
    function setIgnored(list) { localStorage.setItem(STORAGE_KEY_IGNORED, JSON.stringify(list)); }

    function markApplied(jobId) {
        var list = getApplied();
        if (list.indexOf(jobId) === -1) {
            list.push(jobId);
            setApplied(list);
        }
        var row = document.querySelector('tr[data-job-id="' + CSS.escape(jobId) + '"]');
        if (row) {
            row.classList.add('row-done');
            var doneBtn = row.querySelector('.btn-done');
            if (doneBtn) {
                doneBtn.classList.add('marked');
                doneBtn.textContent = '✅ 已投递';
                doneBtn.disabled = true;
            }
            var ignoreBtn = row.querySelector('.btn-ignore');
            if (ignoreBtn) {
                ignoreBtn.disabled = true;
                ignoreBtn.textContent = '已忽略';
            }
        }
    }

    function markIgnored(jobId) {
        var list = getIgnored();
        if (list.indexOf(jobId) === -1) {
            list.push(jobId);
            setIgnored(list);
        }
        var row = document.querySelector('tr[data-job-id="' + CSS.escape(jobId) + '"]');
        if (row) {
            row.classList.add('row-ignored');
            row.style.display = 'none';
            var ignoreBtn = row.querySelector('.btn-ignore');
            if (ignoreBtn) {
                ignoreBtn.disabled = true;
                ignoreBtn.textContent = '已忽略';
            }
            var doneBtn = row.querySelector('.btn-done');
            if (doneBtn) {
                doneBtn.disabled = true;
                doneBtn.textContent = '已忽略';
            }
        }
    }

    function hideAppliedAndIgnored() {
        var applied = getApplied();
        var ignored = getIgnored();
        document.querySelectorAll('tr[data-job-id]').forEach(function(row) {
            var id = row.dataset.jobId;
            if (applied.indexOf(id) !== -1 || ignored.indexOf(id) !== -1) {
                row.style.display = 'none';
            }
        });
    }

    function showModal(modalId, contentContainerId, list) {
        var container = document.getElementById(contentContainerId);
        if (list.length === 0) {
            container.innerHTML = '<p>暂无记录。</p>';
        } else {
            var html = '<table><thead><tr><th>公司</th><th>职位</th><th>薪资</th><th>地区</th></tr></thead><tbody>';
            list.forEach(function(jobId) {
                var row = document.querySelector('tr[data-job-id="' + CSS.escape(jobId) + '"]');
                if (row) {
                    var company = row.getAttribute('data-company') || '未知公司';
                    var title = row.getAttribute('data-title') || '未知职位';
                    var salary = row.getAttribute('data-salary') || '面议';
                    var district = row.getAttribute('data-district') || '未知';
                    html += '<tr><td>' + company + '</td><td>' + title + '</td><td>' + salary + '</td><td>' + district + '</td></tr>';
                } else {
                    html += '<tr><td colspan="4">' + jobId + '</td></tr>';
                }
            });
            html += '</tbody></table>';
            container.innerHTML = html;
        }
        document.getElementById(modalId).classList.add('active');
    }

    function resetAll() {
        if (confirm('确定要清空所有已投递和已忽略记录吗？此操作不可撤销！')) {
            setApplied([]);
            setIgnored([]);
            location.reload();
        }
    }

    function doApply(btn) {
        var jobId = btn.getAttribute('data-job-id');
        var link = btn.getAttribute('data-link');
        var greeting = btn.getAttribute('data-greeting');
        if (btn.classList.contains('marked')) return;

        navigator.clipboard.writeText(greeting).then(function() {
            var toast = document.getElementById('toast');
            toast.textContent = '✅ 招呼语已复制，正在打开链接...';
            toast.style.display = 'block';
            setTimeout(function() { toast.style.display = 'none'; }, 3000);
        }).catch(function() {
            var textarea = document.createElement('textarea');
            textarea.value = greeting;
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
            var toast = document.getElementById('toast');
            toast.textContent = '✅ 招呼语已复制，正在打开链接...';
            toast.style.display = 'block';
            setTimeout(function() { toast.style.display = 'none'; }, 3000);
        });

        markApplied(jobId);
        if (link && link !== '#') { window.open(link, '_blank'); }
    }

    document.addEventListener('DOMContentLoaded', function() {
        document.querySelectorAll('.btn-copy').forEach(function(btn) {
            btn.addEventListener('click', function() {
                var greeting = this.getAttribute('data-greeting');
                navigator.clipboard.writeText(greeting).then(function() {
                    var toast = document.getElementById('toast');
                    toast.textContent = '✅ 已复制招呼语';
                    toast.style.display = 'block';
                    setTimeout(function() { toast.style.display = 'none'; }, 2000);
                }).catch(function() {
                    var textarea = document.createElement('textarea');
                    textarea.value = greeting;
                    document.body.appendChild(textarea);
                    textarea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textarea);
                    var toast = document.getElementById('toast');
                    toast.textContent = '✅ 已复制招呼语';
                    toast.style.display = 'block';
                    setTimeout(function() { toast.style.display = 'none'; }, 2000);
                });
            });
        });

        document.querySelectorAll('.btn-done').forEach(function(btn) {
            btn.addEventListener('click', function() {
                doApply(this);
            });
        });

        document.querySelectorAll('.btn-ignore').forEach(function(btn) {
            btn.addEventListener('click', function() {
                var jobId = this.getAttribute('data-job-id');
                if (this.disabled) return;
                if (confirm('确定对此岗位不感兴趣吗？它将从列表中隐藏。')) {
                    markIgnored(jobId);
                }
            });
        });

        document.getElementById('btnShowApplied').addEventListener('click', function() {
            showModal('appliedModal', 'appliedListContent', getApplied());
        });

        document.getElementById('btnShowIgnored').addEventListener('click', function() {
            showModal('ignoredModal', 'ignoredListContent', getIgnored());
        });

        document.getElementById('modalCloseApplied').addEventListener('click', function() {
            document.getElementById('appliedModal').classList.remove('active');
        });
        document.getElementById('modalCloseIgnored').addEventListener('click', function() {
            document.getElementById('ignoredModal').classList.remove('active');
        });

        document.getElementById('appliedModal').addEventListener('click', function(e) {
            if (e.target === this) this.classList.remove('active');
        });
        document.getElementById('ignoredModal').addEventListener('click', function(e) {
            if (e.target === this) this.classList.remove('active');
        });

        document.getElementById('btnResetApplied').addEventListener('click', resetAll);

        hideAppliedAndIgnored();
    });
})();
</script>
</body>
</html>
"""

with open('apply_helper.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("\n🌐 已生成辅助投递面板：apply_helper.html")
print("双击打开该文件，即可一键复制招呼语并打开链接。")

print("\n" + "=" * 60)
print("✅ AI 投递清单已保存至: apply_ready_list_ai.csv")
print(f"📌 共生成 {len(df_sorted)} 条招呼语")
print("\n💡 下一步：")
print("   1. 打开 apply_helper.html（双击用浏览器打开）")
print("   2. 点击 '📋 一键投递' → 自动复制+打开链接+标记已投递")
print("   3. 点击 '👎 不感兴趣' → 隐藏该岗位")
print("   4. 点击 '📋 已投递' 或 '📋 已忽略' 查看对应列表（模态框）")