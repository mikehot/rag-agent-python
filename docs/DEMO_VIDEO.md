# Demo video: storyboard and recording checklist

Target length **3 to 5 minutes**, one terminal window, synthetic and public data only. Everything on
screen is produced by `scripts/demo.py`, which never prints a token or a password. A full run of the
script takes about 15 seconds of compute; the rest of the time is narration.

What the video has to make a reviewer believe, in order: (1) permissions are enforced by the
knowledge base, not by the model; (2) the agent layer is bounded and cannot be talked into another
identity; (3) the agent does something the single retrieval cannot; (4) the claims are measured and
their limits are stated.

## Before recording (about 10 minutes)

1. **Plug in the power and close other apps.** Hide notifications; use a fresh terminal profile with a
   large font and no prompt that shows your username or paths.
2. **LM Studio:** load `google/gemma-4-26b-a4b-qat` and `text-embedding-bge-m3`. **Enable Thinking must
   be OFF.** If you reload a model from the command line the setting resets, so do it in the app.
3. **Start the knowledge base** (Java repository, one command, about 30 seconds). It refuses to start
   if the models are missing or Thinking is on, and tells you why:

   ```bash
   scripts/eval_stack.sh up
   ```

4. **Start the agent service** (this repository). `decompose` is only needed for scene 5:

   ```bash
   RAG_AGENT_KB_BASE_URL=http://localhost:8080 RAG_AGENT_POLICY=decompose \
     uv run uvicorn rag_agent.main:app --port 8090
   ```

5. **Rehearse once, off camera.** Do not record the first run of the day: the first run of a prompt set
   can differ in wording from later ones (see
   `evaluation/reports/recheck-first-run-effect-local-2026-10-14.md`).

   ```bash
   uv run python scripts/demo.py --credentials "$CREDS"   # $CREDS is printed by `eval_stack.sh up`
   ```

6. **Screen hygiene.** Do not show: the credentials file or its path, the backend log, any
   `Authorization` header, the LM Studio server log. The script is safe, but a stray `ls` or `cat` is
   not.

## Recording

```bash
uv run python scripts/demo.py --credentials "$CREDS" --pause
```

`--pause` waits for Enter between scenes, so you set the pace. Use `--only 2,3` to redo specific
scenes. Do not edit answers out of the recording: if a scene says something unexpected, say so, or
redo the whole take.

| # | Time | On screen | Say (Chinese) | Say (English, one line) |
|---|---|---|---|---|
| 1 | 0:00–0:35 | Both services ready; four actors and how many documents each can see (24 / 2 / 2 / 0) | 同一个知识库，四个人，看到的文档数量不一样。这个限制是在检索的 SQL 里做的，不是靠提示词。 | One knowledge base, four people, four different views; the limit is enforced in the retrieval SQL, not in a prompt. |
| 2 | 0:35–1:10 | SRE asks about Pod restart policy; answer, tool call, 5 chunks, file name written in the answer | 有权限的人得到带出处的回答。这里是 Python 智能体调用 Java 服务的只读工具，用的是这个人自己的身份。 | An authorized user gets a cited answer; the Python agent calls the Java tools with that user's own identity. |
| 3 | 1:10–1:35 | Outsider, same question: 0 chunks retrieved, "not in the knowledge base" | 同一个问题，没有权限的人什么也拿不到：检索到 0 个片段，模型根本没见过这些内容。 | Same question, no grant: zero chunks retrieved, so the model never saw the content. |
| 4 | 1:35–2:00 | SRE asks an HR question; it is declined | 有权限的人问了不属于自己的主题，也是拒答。它仍然检索到 5 个片段（都是 Kubernetes 的），模型判断它们回答不了。 | A permitted user asking about someone else's topic is declined. |
| 5 | 2:00–3:00 | Legal asks a two-part question; Java alone says "no information"; the agent answers both parts using two searches | 这是智能体层存在的理由：Java 单次检索回答不了的跨文档问题，智能体拆成两次检索再合并。 | This is why the agent layer exists: a question that needs two documents, answered with two searches. |
| 6 | 3:00–3:30 | A prompt claiming to be an admin with `userId=admin`; the answer is an ordinary one | 有人在问题里冒充管理员。身份只来自 token，工具的参数里根本没有 userId 这一项。 | Identity comes only from the token; the tool has no userId parameter to fill. |
| 7 | 3:30–4:30 | The evidence and, importantly, what it does not show | 最后是证据，以及它不能证明的部分。我不会把这些数字说成产品指标。 | The evidence, and what it does not prove. |

## Things to say out loud, and not to cut

- **Scenes 2 and 3 can show "轮数 3".** That is the empty-reply retry firing: the model produced an
  empty reply after the tool result and the agent asked once more. It is a real failure mode that was
  found and mitigated, and it is a good thing to point at.
- **Scene 5 shows "回答里写出的文件名：无".** The answer names the laws (《数据安全法》第二十七条), not
  file names, and that field is a strict file-name match. It is a known limitation of the field, not a
  missing citation.
- **The scene 5 question was picked from the evaluated set** because the single-retrieval baseline
  fails it in every run (3 of 3) and decomposition passed it in every run (5 of 5). It is a
  demonstration, not a random draw.
- **Scene 6 is not a proof of security.** The model ignored the claimed identity and answered
  normally; the structural point is that there is nowhere to put a user id.

## What not to claim

- Not "production ready", "enterprise grade", or any figure about users, latency under load, or cost:
  none was measured.
- Not that 43/45 or any single score is a stable property. The Java project's same configuration
  scored 43/45 one day and 38/45 the next with identical retrieval, and the cause is unknown.
- Not that the agent is "better than RAG": it is better on one kind of question, under one prompt,
  with twelve questions written by one person.

## After recording

- Export 3 stills: scene 1 (permissions), scene 3 (zero chunks), scene 5 (two searches). Check each
  for tokens, file paths and usernames before publishing.
- Add the link to the Java repository's README and PORTFOLIO, and tick A5 in its ROADMAP.
- If you publish it, say in the description that the data is public (Kubernetes documentation under
  CC BY 4.0 and Chinese statutes) and that nothing runs in a hosted environment.
