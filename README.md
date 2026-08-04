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
- 重点动态：优先展示超时待回应、我关注、最近更新和已完成事项。
- 回应时限：默认要求 72 小时内首次回应，并与组织天气联动。
- 结果评价：事项完成后，员工可评价已解决、部分解决或未解决。
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

先在 Supabase SQL Editor 执行仓库中的 `supabase_schema.sql`。它会创建规范化的反馈、点赞、结果评价、事项和状态历史表，并保留旧 `stellar_data` 表作为迁移备份。已有项目也可以安全地重新执行该脚本，用于补充新表和索引。

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
RESPONSE_SLA_HOURS = "72"
GMAIL_SENDER_EMAIL = "your-gmail@gmail.com"
GMAIL_APP_PASSWORD = "your-16-character-app-password"
ADMIN_EMAIL = "manager@example.com"
STELLAR_APP_URL = "https://your-streamlit-app.streamlit.app"
```

执行新表结构后，应用会在新表为空时自动复制 `stellar_data/main` 的旧数据；旧数据不会被删除。配置 Supabase 后，反馈会保存到远程数据库，重新 git push 部署不会覆盖已有反馈。

`RESPONSE_SLA_HOURS` 控制首次回应时限，未配置时默认为 72 小时。解决度评价依赖 `stellar_resolution_ratings` 表；运行最新 `supabase_schema.sql` 后自动启用。

配置 Gmail 通知时，请使用 Google 账号开启两步验证后生成的“应用专用密码”，不要填写 Gmail 登录密码。`GMAIL_SENDER_EMAIL` 是发件 Gmail，`ADMIN_EMAIL` 是管理者收件地址，支持用逗号填写多个收件人。`STELLAR_APP_URL` 用于在通知邮件中提供应用入口。邮件发送是提交后的尽力通知：即使 Gmail 暂时不可用，反馈仍会正常保存。
