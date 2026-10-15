# Demo video: storyboard and recording checklist

Target length **3 to 5 minutes**, one terminal window, synthetic and public data only. Everything on
screen is produced by `scripts/demo.py` (part A, users) and `scripts/demo_admin.py` (part B,
administrator), neither of which ever prints a token or a password. Part A takes about 15 seconds of
compute and part B about 7; the rest of the time is narration. Both together run about 7 minutes, so
see "A cut that fits 5 minutes" below.

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

## Part B: upload and administrator grant/revoke (the second driver)

The first driver shows what users see. This one shows what an administrator does and that the answer
follows it immediately. It uploads a new **synthetic** document, asks as a new employee before any
grant, grants READ, asks again, revokes, and asks a third time. It then deletes the document (use
`--keep` to leave it).

```bash
uv run python scripts/demo_admin.py \
  --admin-credentials "$ADMIN" --credentials "$CREDS" --pause     # $ADMIN is printed by `eval_stack.sh up`
```

A full run is about 7 seconds of compute. Like `demo.py`, it never prints a token or a password;
keep the admin credentials file off screen.

| # | Time | On screen | Say (Chinese) | Say (English, one line) |
|---|---|---|---|---|
| B1 | 0:00–0:30 | A new document is uploaded; status becomes `ready`; it was cut into 1 chunk | 管理员上传一份新文档，系统异步解析、切片、向量化，状态变成 ready。 | The admin uploads a document; it is parsed, chunked and embedded, then marked ready. |
| B2 | 0:30–0:55 | The new employee sees 0 documents; asking gets "not in the knowledge base", 0 chunks | 文档已经就绪，但这位员工没有授权，所以什么也看不到，问也问不出来。 | It is indexed, but this employee has no grant, so nothing is visible and nothing is retrievable. |
| B3 | 0:55–1:10 | The grant: `principalType=USER permission=READ` | 管理员给这位员工授予 READ。 | The admin grants READ to this user. |
| B4 | 1:10–1:35 | The employee now sees the document; the answer cites it (600 / 400 yuan per night) | 同一个问题，现在带出处地回答了。 | The same question is now answered with a citation. |
| B5 | 1:35–2:10 | Revoke (HTTP 200); the document disappears; the same question is declined again | 撤销授权，回答立刻跟着变回去。权限是检索查询里的过滤条件，不是模型记住的东西。 | Revoke it and the answer follows at once: permission is a filter in the retrieval query, not something the model remembers. |
| B6 | optional | The demo document is deleted | 清理演示文档。 | Cleanup. |

**Do not claim** that revoking also clears what a *remembered* conversation already contains: this
driver asks without a session. If conversation memory is used, an earlier answer stays in the
conversation. The agent now retrieves again on every remembered turn and does not answer from the
conversation without a retrieval (see the README and the
[enforcement report](../evaluation/reports/enforce-retrieval-local-2026-10-15.md)), but that was
measured on three valid cases and is not shown in the video. Also do not claim anything about
propagation across several backend replicas; this is one local instance.

### A cut that fits 5 minutes

The acceptance text for A5 asks for 3 to 5 minutes. Parts A and B together run about 7. A cut that
covers everything A5 lists (upload, cited answer, refusal, permission denial, admin ACL) in about 5:

| Order | Scene | About |
|---|---|---|
| 1 | A1 permissions per user | 0:30 |
| 2 | B1 upload, B2 before grant, B3 grant, B4 after grant, B5 revoke | 2:00 |
| 3 | A2 cited answer through the agent (shows the tool call) | 0:35 |
| 4 | A5 cross-document question | 0:50 |
| 5 | A7 evidence and its limits | 0:40 |

Drop A3, A4 and A6 (they repeat points B already makes) and B6. Record Part A and Part B as separate
takes and join them.

### Still not covered

The Flutter client and the A1 **device** spot check (post-revoke ask and citation on a phone) are not in
either driver; they need a physical device and are described in `docs/DEMO.md` section 8 of the Java
repository.

## Recording with no voice

If you cannot or do not want to speak, let the screen do the talking. Both drivers take two switches:

| Switch | What it does |
|---|---|
| `--captions` | prints a Chinese and an English explanation (the same sentences as the tables above) before each scene |
| `--auto` | advances by itself after a reading delay and clears the screen at every scene; nobody presses Enter |

```bash
uv run python scripts/demo.py --credentials "$CREDS" --captions --auto
```

```bash
uv run python scripts/demo_admin.py --admin-credentials "$ADMIN" --credentials "$CREDS" --captions --auto
```

Start the recording, type one command, wait, stop. Part A takes about 2.5 minutes at real speed and part
B about 1.5, so together they fit the 5 minutes A5 asks for. Add `--speed 0.2` to rehearse quickly
(every delay is shortened to a fifth). Long results (the cross-document answer, the evidence table) stay
on screen longer than short ones; the per-scene delays are in `scripts/demo_captions.py`.

A silent video is fine for a README or a portfolio page: people often watch with the sound off. The
captions are the same words as the narration, so the recording says what the speaker would have said.

## The recording that exists

A silent 3 min 47 s recording made with these drivers (`--captions --auto`, terminal only, no
narration) is committed to the Java repository as
[`docs/demo/rag-agent-demo.mp4`](https://github.com/mikehot/rag-knowledge-base/blob/main/docs/demo/rag-agent-demo.mp4),
with notes on what it shows and what it does not
([`docs/demo/README.md`](https://github.com/mikehot/rag-knowledge-base/blob/main/docs/demo/README.md)).
It was assembled from three takes (part A without the last scene, a re-recorded last scene with its
final frame held for 12 seconds, and part B).

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
