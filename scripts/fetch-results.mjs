/* ═══════════════════════════════════════════════════════════════════════════
   AUTOMATIC RESULT FETCHER  (GitHub Actions version)
   Runs on a schedule, pulls finished Premier League matches from
   football-data.org, and writes any that are missing into the same Firebase
   database the predictor reads from.

   It only ever FILLS BLANKS. A score you typed in by hand is never
   overwritten, so a manual correction always wins.

   Two repository secrets (Settings > Secrets and variables > Actions):
     FOOTBALL_DATA_TOKEN   your free key from football-data.org/client/register
     FIREBASE_DB_URL       https://your-db.firebaseio.com   (no trailing slash)

   Runs on a schedule from .github/workflows/fetch-results.yml.  You can also
   start it by hand from the Actions tab, with a "dry run" tickbox that makes
   it report what it WOULD do without writing anything.
   ═══════════════════════════════════════════════════════════════════════════ */

const ROOT   = "pl2627";
const SEASON = 2026;                 // the 2026-27 campaign

/* our three-letter codes, keyed by a squashed version of the club name */
const CODES = {
  arsenal:"ARS", astonvilla:"AVL", bournemouth:"BOU", brentford:"BRE",
  brightonhovealbion:"BHA", chelsea:"CHE", coventrycity:"COV", crystalpalace:"CRY",
  everton:"EVE", fulham:"FUL", hullcity:"HUL", ipswichtown:"IPS",
  leedsunited:"LEE", liverpool:"LIV", manchestercity:"MCI", manchesterunited:"MUN",
  newcastleunited:"NEW", nottinghamforest:"NFO", sunderland:"SUN", tottenhamhotspur:"TOT"
};

/* Short forms and nicknames a feed might use instead of the full club name.
   Deliberately excludes ambiguous ones — "City", "United", "Reds" and
   "Blues" could each mean more than one club in this league. */
const ALIASES = {
  gunners:"ARS",
  villa:"AVL", avfc:"AVL",
  cherries:"BOU", afcbournemouth:"BOU",
  bees:"BRE",
  brighton:"BHA", brightonhove:"BHA", brightonandhovealbion:"BHA", seagulls:"BHA",
  cfc:"CHE", blues:"CHE",
  coventry:"COV", skyblues:"COV", skyblue:"COV",
  palace:"CRY", cpfc:"CRY", eagles:"CRY",
  toffees:"EVE",
  cottagers:"FUL",
  hull:"HUL", tigers:"HUL",
  ipswich:"IPS", tractorboys:"IPS",
  leeds:"LEE", lufc:"LEE", whites:"LEE",
  lfc:"LIV",
  mancity:"MCI", manchestercityfc:"MCI", mcfc:"MCI", citizens:"MCI",
  manutd:"MUN", manunited:"MUN", manchesterutd:"MUN", mufc:"MUN", manu:"MUN", redevils:"MUN",
  newcastle:"NEW", nufc:"NEW", magpies:"NEW", toon:"NEW",
  forest:"NFO", nottsforest:"NFO", nottmforest:"NFO", nottingham:"NFO", nffc:"NFO",
  blackcats:"SUN", safc:"SUN",
  spurs:"TOT", tottenham:"TOT", thfc:"TOT"
};

/* "Brighton & Hove Albion FC" -> "brightonhovealbion", "Man Utd" -> "manutd" */
function squash(name){
  return String(name || "")
    .toLowerCase()
    .normalize("NFD").replace(/[\u0300-\u036f]/g, "")
    .replace(/&/g, "")
    .replace(/[^a-z ]/g, " ")
    .replace(/\butd\b/g, "united")
    .replace(/\b(fc|afc|association|football|club)\b/g, "")
    .replace(/\band\b/g, "")
    .replace(/\s+/g, "")
    .trim();
}

/* Tries, in order: the exact club name, a known short form, then a unique
   prefix. Anything ambiguous ("Manchester" on its own) returns null rather
   than guessing, so it lands in couldNotMap instead of a wrong fixture. */
function codeFor(team){
  if (!team) return null;
  const fields = [team.shortName, team.name, team.tla].filter(Boolean);

  for (const f of fields){
    const k = squash(f);
    if (k && CODES[k]) return CODES[k];
  }
  for (const f of fields){
    const k = squash(f);
    if (k && ALIASES[k]) return ALIASES[k];
  }
  for (const f of fields){
    const k = squash(f);
    if (!k || k.length < 3) continue;
    const hits = [...new Set(
      Object.keys(CODES).filter(c => c.startsWith(k) || k.startsWith(c)).map(c => CODES[c])
    )];
    if (hits.length === 1) return hits[0];
  }
  return null;
}

/* the predictor stores results under `{matchday}_{HOME}_{AWAY}` */
function matchId(m){
  const h = codeFor(m.homeTeam), a = codeFor(m.awayTeam);
  if (!h || !a || !m.matchday) return null;
  return `${m.matchday}_${h}_${a}`;
}

function buildUpdates(apiMatches, existing){
  const writes = {}, skipped = [], unmapped = [];
  for (const m of apiMatches){
    if (m.status !== "FINISHED") continue;
    const ft = m.score && m.score.fullTime;
    if (!ft || ft.home === null || ft.away === null) continue;
    const id = matchId(m);
    if (!id){
      unmapped.push(`${m.homeTeam && m.homeTeam.name} v ${m.awayTeam && m.awayTeam.name}`);
      continue;
    }
    if (existing[id]){ skipped.push(id); continue; }        // never overwrite
    writes[id] = { h: ft.home, a: ft.away, at: Date.now(), src: "auto" };
  }
  return { writes, skipped, unmapped };
}

const say = (...a) => console.log("[fetch-results]", ...a);

async function main(){
  const token = process.env.FOOTBALL_DATA_TOKEN;
  const dbUrl = (process.env.FIREBASE_DB_URL || "").replace(/\/$/, "");
  const dry   = process.env.DRY_RUN === "1" || process.env.DRY_RUN === "true";

  if (!token || !dbUrl){
    say("MISSING CONFIG — add the FOOTBALL_DATA_TOKEN and FIREBASE_DB_URL secrets");
    process.exit(1);
  }

  say(dry ? "DRY RUN — nothing will be written" : "live run");

  const api = await fetch(
    `https://api.football-data.org/v4/competitions/PL/matches?season=${SEASON}&status=FINISHED`,
    { headers: { "X-Auth-Token": token } });
  if (!api.ok){
    say(`FOOTBALL-DATA ERROR ${api.status}` +
        (api.status === 403 ? " — token rejected, check the FOOTBALL_DATA_TOKEN secret" :
         api.status === 429 ? " — rate limited, it will retry on the next run" : ""));
    process.exit(1);
  }
  const data = await api.json();
  say(`football-data returned ${(data.matches || []).length} finished match(es)`);

  const cur = await fetch(`${dbUrl}/${ROOT}/results.json`);
  if (!cur.ok){
    say(`FIREBASE READ ERROR ${cur.status} — check the FIREBASE_DB_URL secret`);
    process.exit(1);
  }
  const existing = (await cur.json()) || {};
  say(`database already holds ${Object.keys(existing).length} result(s)`);

  const { writes, skipped, unmapped } = buildUpdates(data.matches || [], existing);
  const ids = Object.keys(writes);

  if (unmapped.length) say("COULD NOT MAP (tell Claude these names):", unmapped.join(" | "));
  else say("could not map: none");
  say(`already had: ${skipped.length}`);
  say(`to add: ${ids.length}`);
  ids.forEach(id => say(`   ${id} = ${writes[id].h}-${writes[id].a}`));

  if (dry){
    say("DRY RUN complete — nothing written");
    return;
  }
  if (!ids.length){
    say("nothing new to write");
    return;
  }
  const put = await fetch(`${dbUrl}/${ROOT}/results.json`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(writes)
  });
  if (!put.ok){
    say(`FIREBASE WRITE ERROR ${put.status}`);
    process.exit(1);
  }
  say(`WROTE ${ids.length} result(s) — points are live`);
}

main().catch(e => { say("UNEXPECTED ERROR:", e && e.message || e); process.exit(1); });
