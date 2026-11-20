import json, urllib.request, pathlib
cfg = json.loads(pathlib.Path.home().joinpath(".bailian/config.json").read_text(encoding="utf-8"))
prof = cfg[cfg["active_config"]]
key, base = prof["api_key"], prof["base_url"].rstrip("/")
body = {"model": prof.get("default_text_model","qwen3.8-max"),
        "messages":[{"role":"user","content":"只输出一个字母。中国的首都是哪里？A. 上海 B. 北京 C. 广州"}],
        "max_tokens":1,"temperature":0,"logprobs":True,"top_logprobs":5}
for path in ["/v1/chat/completions","/compatible-mode/v1/chat/completions"]:
    url = base + path
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
        headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read())
        ch = d["choices"][0]
        lp = ch.get("logprobs")
        print("OK", url, "| model:", d.get("model"), "| content:", repr(ch["message"]["content"]))
        if lp and lp.get("content"):
            print("top_logprobs:", [(t["token"], round(t["logprob"],3)) for t in lp["content"][0]["top_logprobs"][:6]])
        else:
            print("logprobs: NONE")
        break
    except Exception as e:
        msg = str(e)
        try: msg += " | " + e.read().decode()[:300]
        except Exception: pass
        print("FAIL", url, "|", msg[:400])

