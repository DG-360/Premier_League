/* ═══════════════════════════════════════════════════════════════════════════
   AUTOMATIC RESULT FETCHER — GitHub Actions version

   Purpose:
   - Pull Premier League match results from football-data.org.
   - Keep only matches whose status is FINISHED.
   - Match them to the fixture IDs used by the predictor.
   - Write missing results to Firebase.
   - Never overwrite a score already stored in Firebase.

   Environment variables supplied by GitHub Actions:
     FOOTBALL_DATA_TOKEN
     FIREBASE_DB_URL
     DRY_RUN

   The GitHub repository secrets themselves can have different names
   (for example TK1 and TK2). The workflow YAML maps those secrets to
   the environment-variable names above.
   ═══════════════════════════════════════════════════════════════════════════ */

const ROOT = "pl2627";


/* ═══════════════════════════════════════════════════════════════════════════
   TEAM CODE MAPPING
   ═══════════════════════════════════════════════════════════════════════════ */

const CODES = {
  arsenal: "ARS",
  astonvilla: "AVL",
  bournemouth: "BOU",
  brentford: "BRE",
  brightonhovealbion: "BHA",
  chelsea: "CHE",
  coventrycity: "COV",
  crystalpalace: "CRY",
  everton: "EVE",
  fulham: "FUL",
  hullcity: "HUL",
  ipswichtown: "IPS",
  leedsunited: "LEE",
  liverpool: "LIV",
  manchestercity: "MCI",
  manchesterunited: "MUN",
  newcastleunited: "NEW",
  nottinghamforest: "NFO",
  sunderland: "SUN",
  tottenhamhotspur: "TOT"
};


/* Short forms / nicknames that the API might use. */
const ALIASES = {
  gunners: "ARS",

  villa: "AVL",
  avfc: "AVL",

  cherries: "BOU",
  afcbournemouth: "BOU",

  bees: "BRE",

  brighton: "BHA",
  brightonhove: "BHA",
  brightonandhovealbion: "BHA",
  seagulls: "BHA",

  cfc: "CHE",
  blues: "CHE",

  coventry: "COV",
  skyblues: "COV",
  skyblue: "COV",

  palace: "CRY",
  cpfc: "CRY",
  eagles: "CRY",

  toffees: "EVE",

  cottagers: "FUL",

  hull: "HUL",
  tigers: "HUL",

  ipswich: "IPS",
  tractorboys: "IPS",

  leeds: "LEE",
  lufc: "LEE",
  whites: "LEE",

  lfc: "LIV",

  mancity: "MCI",
  manchestercityfc: "MCI",
  mcfc: "MCI",
  citizens: "MCI",

  manutd: "MUN",
  manunited: "MUN",
  manchesterutd: "MUN",
  mufc: "MUN",
  manu: "MUN",
  redevils: "MUN",

  newcastle: "NEW",
  nufc: "NEW",
  magpies: "NEW",
  toon: "NEW",

  forest: "NFO",
  nottsforest: "NFO",
  nottmforest: "NFO",
  nottingham: "NFO",
  nffc: "NFO",

  blackcats: "SUN",
  safc: "SUN",

  spurs: "TOT",
  tottenham: "TOT",
  thfc: "TOT"
};


/* Convert a team name into a simple comparison key.

   Examples:
   "Brighton & Hove Albion FC" -> "brightonhovealbion"
   "Manchester United FC"      -> "manchesterunited"
*/
function squash(name) {
  return String(name || "")
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/&/g, "")
    .replace(/[^a-z ]/g, " ")
    .replace(/\butd\b/g, "united")
    .replace(/\b(fc|afc|association|football|club)\b/g, "")
    .replace(/\band\b/g, "")
    .replace(/\s+/g, "")
    .trim();
}


/* Convert a football-data.org team object into our 3-letter code.

   Tries:
   1. Exact known name
   2. Known alias
   3. Unique prefix match

   If the result would be ambiguous, return null instead of guessing.
*/
function codeFor(team) {
  if (!team) return null;

  const fields = [
    team.shortName,
    team.name,
    team.tla
  ].filter(Boolean);


  /* Exact name */
  for (const field of fields) {
    const key = squash(field);

    if (key && CODES[key]) {
      return CODES[key];
    }
  }


  /* Alias */
  for (const field of fields) {
    const key = squash(field);

    if (key && ALIASES[key]) {
      return ALIASES[key];
    }
  }


  /* Unique prefix */
  for (const field of fields) {
    const key = squash(field);

    if (!key || key.length < 3) {
      continue;
    }

    const hits = [
      ...new Set(
        Object.keys(CODES)
          .filter(
            candidate =>
              candidate.startsWith(key) ||
              key.startsWith(candidate)
          )
          .map(candidate => CODES[candidate])
      )
    ];

    if (hits.length === 1) {
      return hits[0];
    }
  }


  return null;
}


/* Predictor result IDs have this format:

     {matchday}_{HOME}_{AWAY}

   Example:
     1_LIV_ARS
*/
function matchId(match) {
  const home = codeFor(match.homeTeam);
  const away = codeFor(match.awayTeam);

  if (!home || !away || !match.matchday) {
    return null;
  }

  return `${match.matchday}_${home}_${away}`;
}


/* Compare API results against results already in Firebase.

   Important:
   Existing Firebase results are NEVER overwritten.
*/
function buildUpdates(apiMatches, existing) {
  const writes = {};
  const skipped = [];
  const unmapped = [];

  for (const match of apiMatches) {

    /* Ignore anything that is not finished */
    if (match.status !== "FINISHED") {
      continue;
    }


    const fullTime =
      match.score &&
      match.score.fullTime;


    /* Ignore a finished match if the API somehow has no score */
    if (
      !fullTime ||
      fullTime.home === null ||
      fullTime.home === undefined ||
      fullTime.away === null ||
      fullTime.away === undefined
    ) {
      continue;
    }


    const id = matchId(match);


    /* Couldn't translate team names into our fixture ID */
    if (!id) {
      unmapped.push(
        `${match.homeTeam?.name || "Unknown home team"} v ` +
        `${match.awayTeam?.name || "Unknown away team"}`
      );

      continue;
    }


    /* Never overwrite an existing result */
    if (existing[id]) {
      skipped.push(id);
      continue;
    }


    writes[id] = {
      h: fullTime.home,
      a: fullTime.away,
      at: Date.now(),
      src: "auto"
    };
  }


  return {
    writes,
    skipped,
    unmapped
  };
}


/* Consistent log prefix */
const say = (...args) => {
  console.log("[fetch-results]", ...args);
};


/* ═══════════════════════════════════════════════════════════════════════════
   MAIN
   ═══════════════════════════════════════════════════════════════════════════ */

async function main() {

  const token =
    (process.env.FOOTBALL_DATA_TOKEN || "").trim();

  const dbUrl =
    (process.env.FIREBASE_DB_URL || "")
      .trim()
      .replace(/\/$/, "");

  const dry =
    process.env.DRY_RUN === "1" ||
    process.env.DRY_RUN === "true";


  /* ───────────────────────────────────────────────────────────────────────
     Configuration validation
     ─────────────────────────────────────────────────────────────────────── */

  if (!token) {
    say("MISSING CONFIG — FOOTBALL_DATA_TOKEN is empty");
    process.exit(1);
  }


  if (!dbUrl) {
    say("MISSING CONFIG — FIREBASE_DB_URL is empty");
    process.exit(1);
  }


  if (!dbUrl.startsWith("https://")) {
    say("INVALID CONFIG — FIREBASE_DB_URL must begin with https://");
    process.exit(1);
  }


  say(dry
    ? "DRY RUN — nothing will be written"
    : "LIVE RUN"
  );


  /* ───────────────────────────────────────────────────────────────────────
     STEP 1 — Read Premier League matches from football-data.org

     No season/status query is used here.

     football-data.org defaults this competition endpoint to the
     current season. We filter FINISHED matches ourselves below.
     ─────────────────────────────────────────────────────────────────────── */

  const footballUrl =
    "https://api.football-data.org/v4/competitions/PL/matches";


  say("requesting Premier League matches from football-data.org");


  const api = await fetch(
    footballUrl,
    {
      method: "GET",

      headers: {
        "X-Auth-Token": token,
        "Accept": "application/json"
      }
    }
  );


  /* Print the API's real error message if something goes wrong.

     This is important because "HTTP 400" by itself does not explain
     whether the problem is the token, competition, parameters, etc.
  */
  if (!api.ok) {

    const responseText =
      await api.text().catch(() => "");


    say(`FOOTBALL-DATA ERROR ${api.status}`);


    if (responseText) {
      say(`football-data response: ${responseText}`);
    }


    if (api.status === 400) {
      say(
        "The football-data request was rejected as invalid. " +
        "Check the response message above."
      );
    }


    if (api.status === 401) {
      say(
        "Authentication failed — check the TK2 football-data.org token."
      );
    }


    if (api.status === 403) {
      say(
        "Access denied — check the football-data.org token/account permissions."
      );
    }


    if (api.status === 429) {
      say(
        "Rate limited — the next scheduled run will retry automatically."
      );
    }


    process.exit(1);
  }


  const data = await api.json();


  const allMatches =
    Array.isArray(data.matches)
      ? data.matches
      : [];


  const finishedMatches =
    allMatches.filter(
      match => match.status === "FINISHED"
    );


  say(
    `football-data returned ${allMatches.length} total match(es)`
  );

  say(
    `${finishedMatches.length} match(es) are FINISHED`
  );


  /* ───────────────────────────────────────────────────────────────────────
     STEP 2 — Read existing results from Firebase
     ─────────────────────────────────────────────────────────────────────── */

  const firebaseResultsUrl =
    `${dbUrl}/${ROOT}/results.json`;


  const currentResponse =
    await fetch(firebaseResultsUrl);


  if (!currentResponse.ok) {

    const responseText =
      await currentResponse.text().catch(() => "");


    say(
      `FIREBASE READ ERROR ${currentResponse.status}`
    );


    if (responseText) {
      say(`Firebase response: ${responseText}`);
    }


    process.exit(1);
  }


  const existing =
    (await currentResponse.json()) || {};


  say(
    `database already holds ${Object.keys(existing).length} result(s)`
  );


  /* ───────────────────────────────────────────────────────────────────────
     STEP 3 — Determine which scores are missing
     ─────────────────────────────────────────────────────────────────────── */

  const {
    writes,
    skipped,
    unmapped
  } = buildUpdates(
    finishedMatches,
    existing
  );


  const ids =
    Object.keys(writes);


  if (unmapped.length) {

    say(
      "COULD NOT MAP (check these team names):",
      unmapped.join(" | ")
    );

  } else {

    say("could not map: none");
  }


  say(`already had: ${skipped.length}`);

  say(`to add: ${ids.length}`);


  for (const id of ids) {

    say(
      `   ${id} = ${writes[id].h}-${writes[id].a}`
    );
  }


  /* ───────────────────────────────────────────────────────────────────────
     STEP 4 — Dry run stops here
     ─────────────────────────────────────────────────────────────────────── */

  if (dry) {

    say("DRY RUN complete — nothing written");

    return;
  }


  /* Nothing new */
  if (!ids.length) {

    say("nothing new to write");

    return;
  }


  /* ───────────────────────────────────────────────────────────────────────
     STEP 5 — Write missing scores to Firebase
     ─────────────────────────────────────────────────────────────────────── */

  const put =
    await fetch(
      firebaseResultsUrl,
      {
        method: "PATCH",

        headers: {
          "Content-Type": "application/json"
        },

        body: JSON.stringify(writes)
      }
    );


  if (!put.ok) {

    const responseText =
      await put.text().catch(() => "");


    say(`FIREBASE WRITE ERROR ${put.status}`);


    if (responseText) {
      say(`Firebase response: ${responseText}`);
    }


    process.exit(1);
  }


  say(
    `WROTE ${ids.length} result(s) — points are live`
  );
}


/* Actually execute the fetcher when GitHub Actions runs:

     node scripts/fetch-results.mjs
*/
main().catch(error => {

  say(
    "UNEXPECTED ERROR:",
    error?.message || String(error)
  );

  process.exit(1);
});
