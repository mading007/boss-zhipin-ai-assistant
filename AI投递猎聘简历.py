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
    'years_exp': 1.0,
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

# ---------- 自动查找最新的猎聘数据文件 ----------
# 搜索 D 盘 scripts 目录下的猎聘数据文件，排除 _with_jd 文件（那些是带JD的详情）
scripts_dir = r'D:\Desktop\boss-zhipin-scraper-master\scripts'
pattern = os.path.join(scripts_dir, 'liepin_api_*.json')
all_files = glob.glob(pattern)
# 排除 _with_jd 文件
candidate_files = [f for f in all_files if '_with_jd' not in f]
if not candidate_files:
    raise FileNotFoundError(f"❌ 在 {scripts_dir} 下未找到 liepin_api_*.json 文件，请先运行猎聘爬虫")

# 按修改时间取最新
latest_file = max(candidate_files, key=os.path.getmtime)
print(f"📂 使用猎聘数据文件: {os.path.basename(latest_file)}")

with open(latest_file, 'r', encoding='utf-8') as f:
    jobs = json.load(f)
df = pd.DataFrame(jobs)
print(f"📂 加载 {len(df)} 条猎聘岗位数据")

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
    match = re.search(r'(\d+\.?\d*)-(\d+\.?\d*)K', salary_str, re.IGNORECASE)
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

    # 注意：猎聘数据可能没有 jd 字段，所以语义匹配用 title + tags 代替
    jd_text = row.get('jd', '') or row.get('title', '') + ' ' + row.get('tags', '')
    my_text = MY_PROFILE.get('personal_summary', '')
    if jd_text and my_text and len(jd_text) > 20:
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

# ---------- 招呼语生成 ----------
def generate_greeting_with_ai(row):
    if not DEEPSEEK_CONFIG['api_key'] or DEEPSEEK_CONFIG['api_key'] == 'sk-你的DeepSeek API密钥':
        return _fallback_greeting(row)

    title = row.get('title', '该岗位')
    jd_text = row.get('jd', '') or row.get('title', '') + ' ' + row.get('tags', '')
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

    grad_year = "2026届"
    prompt = f"""请根据以下信息，为求职者撰写一段发给HR的打招呼语（80-120字）：

【求职者背景】
{my_summary}
掌握技术：{tech_str}
工作年限：{MY_PROFILE['years_exp']}年
学历：本科

【目标岗位】
职位：{title}
岗位描述：{jd_text}

【要求】
1. 称呼统一用"您好"，不要用"尊敬的XXX"。
2. 第一句直接说："我是{MY_PROFILE['your_name']}，对{title}岗位很感兴趣。"
3. 第二句：用1句话讲一个具体的项目成果，不要说"我熟悉Python"这种空话。
4. 第三句：表达"我能干活"的意思，用"希望能有面试机会进一步交流"或"期待有机会聊聊"。
5. 不要刻意出现"GitHub""开源""代码可查"等词。
6. 不要过度强调应届生身份，除非JD里明确写了"欢迎应届生"。
7. 语气：{style_desc}。
8. 直接输出招呼语正文，不要加任何额外说明。

招呼语："""

    try:
        headers = {
            'Authorization': f'Bearer {DEEPSEEK_CONFIG["api_key"]}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': DEEPSEEK_CONFIG['model'],
            'messages': [
                {'role': 'system', 'content': '你是一位专业的求职顾问，擅长帮求职者写出真诚、有吸引力的打招呼语。'},
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
    return f"您好，我是{name}，对{title}岗位很感兴趣。我熟悉{tech_str}，有{MY_PROFILE['years_exp']}年经验，希望能有机会进一步交流。"

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
df_sorted[output_cols].to_csv('apply_ready_list_liepin.csv', index=False, encoding='utf-8-sig')

# ---------- HTML面板 ----------
html_content = """
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>猎聘 投递助手</title>
<style>
    body { font-family: 'Segoe UI', sans-serif; background: #f5f7fa; padding: 20px; }
    table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #e0e4e8; }
    th { background: #2c3e50; color: white; font-weight: 600; }
    tr:hover { background: #f1f5f9; }
    .match-score { font-weight: bold; color: #2c3e50; }
    .btn-copy { background: #3498db; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; }
    .btn-copy:hover { background: #2980b9; }
    .btn-link { background: #27ae60; color: white; padding: 6px 12px; border-radius: 4px; text-decoration: none; font-size: 14px; }
    .btn-link:hover { background: #1e8449; }
    .toast { position: fixed; bottom: 20px; right: 20px; background: #2ecc71; color: white; padding: 10px 20px; border-radius: 6px; display: none; }
    .greeting-preview { font-size: 13px; color: #555; max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: inline-block; }
</style>
</head>
<body>
<h2>📋 猎聘 AI 匹配投递清单</h2>
<p>按匹配分从高到低排序，点击"复制招呼语"按钮，再点击岗位链接，到猎聘粘贴发送。</p>
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

    html_content += f"""
    <tr>
        <td class="match-score">{score:.1f}</td>
        <td>{boss_name}</td>
        <td>{title}</td>
        <td>{salary}</td>
        <td>{district}</td>
        <td>
            <button class="btn-copy" data-greeting="{greeting}" onclick="copyGreeting(this)">📋 复制招呼语</button>
            <a href="{link}" target="_blank" class="btn-link">🔗 打开链接</a>
            <span class="greeting-preview" title="{greeting}">{preview}</span>
        </td>
    </tr>
    """

html_content += """
</tbody></table>
<div id="toast" class="toast">✅ 招呼语已复制，快去粘贴吧！</div>
<script>
function copyGreeting(btn) {
    const greeting = btn.getAttribute('data-greeting');
    navigator.clipboard.writeText(greeting).then(() => {
        const toast = document.getElementById('toast');
        toast.style.display = 'block';
        setTimeout(() => { toast.style.display = 'none'; }, 2000);
    }).catch(() => {
        const textarea = document.createElement('textarea');
        textarea.value = greeting;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        const toast = document.getElementById('toast');
        toast.style.display = 'block';
        setTimeout(() => { toast.style.display = 'none'; }, 2000);
    });
}
</script>
</body>
</html>
"""

with open('apply_helper_liepin.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("\n🌐 已生成猎聘辅助投递面板：apply_helper_liepin.html")
print("双击打开该文件，即可一键复制招呼语并打开链接。")

print("\n" + "=" * 60)
print("✅ 猎聘投递清单已保存至: apply_ready_list_liepin.csv")
print(f"📌 共生成 {len(df_sorted)} 条招呼语")
print("\n💡 下一步：")
print("   1. 打开 apply_helper_liepin.html（双击用浏览器打开）")
print("   2. 点击'复制招呼语' → 点击'打开链接' → 在猎聘粘贴发送")
print("   3. 按匹配分从高到低依次投递，每天 10-15 家效果最佳。")