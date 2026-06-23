# Stellar

一个轻量的内部反馈收集与进度公开工具。

## 运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

数据默认保存在 `pulse_data.json`。

## 功能

- 提交反馈：员工提交问题、建议或活动意见。
- AI 整理：可选 DeepSeek、Gemini 或本地规则，把原始表达整理成清晰建议。
- 查看进度：公开查看已提交反馈和事项处理状态。
- 星空意见图：用夜晚富士山背景展示反馈，每颗星代表一条意见。

## AI 配置

推荐 DeepSeek：

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export DEEPSEEK_MODEL="deepseek-v4-flash"
```

也可以使用 Gemini：

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export GEMINI_MODEL="gemini-2.5-flash"
```

或复制 `.streamlit/secrets.example.toml` 为 `.streamlit/secrets.toml` 后填入真实 key。未配置 API 时，应用会使用本地规则，不影响提交反馈。

## 数据持久化

Streamlit 一键部署的本地文件不适合保存正式反馈，重新部署或应用重启可能丢失运行时写入的数据。建议使用 Supabase。

创建表：

```sql
create table if not exists stellar_data (
  id text primary key,
  data jsonb not null,
  updated_at timestamptz default now()
);
```

在 Streamlit Cloud 的 App settings -> Secrets 添加：

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "your-service-role-key"
DEEPSEEK_API_KEY = "your-deepseek-api-key"
DEEPSEEK_MODEL = "deepseek-v4-flash"
```

配置 Supabase 后，反馈会保存到远程数据库，重新 git push 部署不会覆盖已有反馈。
