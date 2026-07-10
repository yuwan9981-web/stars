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
- 相似议题：提交前识别可能重复的反馈，可直接加入已有议题。
- 共鸣空间：星空意见图、意见星座与回声墙集中在侧边栏。
- 富士山组织天气：入口页根据回应率、高热反馈和闭环情况改变天气状态。
- 管理模式：管理层发布正式回应、指定负责人并留下状态时间线。

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

先在 Supabase SQL Editor 执行仓库中的 `supabase_schema.sql`。它会创建规范化的反馈、点赞、事项和状态历史表，并保留旧 `stellar_data` 表作为迁移备份。

旧版单行 JSON 表仍可兼容：

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
ADMIN_PASSWORD = "your-admin-password"
ADMIN_NAME = "管理层"
DELETE_CODE_SALT = "a-long-random-secret"
```

执行新表结构后，应用会在新表为空时自动复制 `stellar_data/main` 的旧数据；旧数据不会被删除。配置 Supabase 后，反馈会保存到远程数据库，重新 git push 部署不会覆盖已有反馈。
