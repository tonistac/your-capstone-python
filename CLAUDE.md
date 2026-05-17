# CJ's Project — Python Agent
 
## What This Does
This is a  research assistant agent  
 
## Architecture
This agent follows a 4-step loop per request:
1. Load conversation history from Supabase (last N turns)
2. Search Pinecone for relevant context documents
3. Run main reasoning with Claude Sonnet (tools: web_search, web_fetch,
   pinecone_search, summarise)
4. Judge the response quality with Claude Haiku, save to Supabase
 
## File Map
- agent.py      → Main agent loop. Do not add tools here.
- tools.py      → All tool definitions and executor functions.
- memory.py     → All Supabase read/write. No agent logic here.
- evals.py      → Judge subagent and eval harness. Keep separate.
- main.py       → CLI only. No business logic.
 
## Environment Variables Required
ANTHROPIC_API_KEY, PINECONE_API_KEY, PINECONE_INDEX,
SUPABASE_URL, SUPABASE_KEY
 
## Conventions
- All tool executors return str. Never return None.
- All errors are caught and returned as "ERROR: <message>" strings.
- Max 10 iterations in agent loop. Log a warning if hit.
- Session IDs are always strings. Never integers.
 
## Do NOT
- Add print statements to production code — use logging
- Import from agent.py in tools.py (circular import)
- Commit .env or any file containing API keys
- Use raw SQL — use supabase-py client methods only
