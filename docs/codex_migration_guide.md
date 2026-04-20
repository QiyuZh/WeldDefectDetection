# Codex 本地记录迁移说明

本文档说明如何把一台电脑上的 Codex 桌面端本地记录迁移到另一台电脑。

## 1. 先区分两类内容

- 项目代码、模型、配置：优先用 `git` / GitHub 迁移。
- Codex 桌面端本地记录：主要在 `%USERPROFILE%\.codex\` 下，需要手动备份和恢复。

## 2. 推荐迁移哪些内容

建议迁移：

- `sessions/`
- `memories/`
- `sqlite/`
- `state_5.sqlite*`
- `logs_2.sqlite*`
- `config.toml`
- `skills/`
- `plugins/`
- `rules/`
- `.codex-global-state.json`
- `AGENTS.md`

通常不建议迁移：

- `cache/`
- `tmp/`
- `.tmp/`
- `.sandbox/`
- `.sandbox-bin/`
- `.sandbox-secrets/`
- `vendor_imports/`

认证相关建议重新登录，不建议直接迁移：

- `auth.json`
- `cap_sid`
- `installation_id`

## 3. 迁移前准备

两台电脑都先退出 Codex 桌面端，再进行备份和恢复。

如果你想最大限度降低兼容风险，建议新旧电脑使用接近版本的 Codex。

## 4. 旧电脑备份

下面命令会把 `%USERPROFILE%\.codex\` 里建议保留的内容复制到桌面上的 `codex-backup`，并打成 zip。

```powershell
$src = "$env:USERPROFILE\.codex"
$bak = "$env:USERPROFILE\Desktop\codex-backup"

Remove-Item -Recurse -Force -LiteralPath $bak -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $bak | Out-Null

robocopy $src $bak /E `
  /XD cache tmp .tmp .sandbox .sandbox-bin .sandbox-secrets vendor_imports `
  /XF auth.json cap_sid installation_id

Compress-Archive -Path "$bak\*" -DestinationPath "$env:USERPROFILE\Desktop\codex-backup.zip" -Force
```

## 5. 新电脑恢复

先在新电脑安装并启动过一次 Codex，确认 `%USERPROFILE%\.codex\` 已经生成，再退出 Codex。

然后执行：

```powershell
$zip = "$env:USERPROFILE\Desktop\codex-backup.zip"
$tmp = "$env:TEMP\codex-backup"
$dst = "$env:USERPROFILE\.codex"

Remove-Item -Recurse -Force -LiteralPath $tmp -ErrorAction SilentlyContinue
Expand-Archive -Path $zip -DestinationPath $tmp -Force
robocopy $tmp $dst /E
```

恢复后重新打开 Codex，并在需要时重新登录账号。

## 6. 最稳的迁移策略

如果你只关心“历史会话、记忆、配置”而不关心登录状态，推荐策略是：

1. 迁移 `sessions/`
2. 迁移 `memories/`
3. 迁移 `sqlite/` 和 `state_5.sqlite*`
4. 迁移 `config.toml`
5. 在新电脑重新登录

这样风险最低。

## 7. 验证是否迁移成功

迁移后可以检查：

- 历史会话是否还能看到
- 之前的记忆是否还在
- 自定义 `skills/`、`plugins/`、`rules/` 是否可用
- 配置项是否延续

如果只是认证失效，但会话和配置都在，这通常是正常的，重新登录即可。

## 8. 常见问题

### 8.1 新电脑看不到旧会话

优先检查：

- 是否真的复制到了 `%USERPROFILE%\.codex\`
- 恢复时 Codex 是否已退出
- 是否遗漏了 `sessions/`、`sqlite/`、`state_5.sqlite*`

### 8.2 登录状态没有保留

这是预期内情况。建议重新登录，不要强依赖迁移 `auth.json`。

### 8.3 迁移后行为异常

可以先保留：

- `sessions/`
- `memories/`
- `config.toml`

然后临时移开下面这些数据库文件，再重新打开 Codex：

- `state_5.sqlite`
- `state_5.sqlite-shm`
- `state_5.sqlite-wal`
- `logs_2.sqlite`
- `logs_2.sqlite-shm`
- `logs_2.sqlite-wal`

这属于“保会话和配置，放弃部分运行态状态”的降级恢复方式。

## 9. 最终建议

最稳的做法是双轨：

- 代码和项目文件走 GitHub
- Codex 本地记录走 `%USERPROFILE%\.codex\` 备份恢复

这样换电脑时，项目和本地使用痕迹都能最大程度保留下来。
