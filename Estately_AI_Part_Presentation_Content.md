# Estately — AI System Section (Slide Content)
### For: AI Engineer's part of the graduation discussion
### Context: comes after the team has already covered motivation/problem/survey at a high level — your part opens with a quick bridge, restates the gap on its own slide, then dives deep into the system that closes it, ending in trust and business payoff

---

## Slide 1 — Title & Bridge

- Estately AI System — The Intelligent Layer Behind the Platform
- [Your Name] — AI Engineer
- One-line bridge: as you've heard, trust and discoverability were two of the biggest gaps we found — here's the system built to close them

---

## Slide 2 — The Gap That Shaped This System

- **88%** of users said they don't fully trust the property listings they see online — a generic AI that just sounds confident wouldn't fix that, it could make it worse
- **42%** named an AI chatbot as the single most-requested feature on a new platform
- **42%** said finding the *right* property is genuinely hard with normal listing sites
- **60%** are just exploring prices and options — not ready to commit to a rigid, multi-field search form

**→ This is the gap the system you're about to see was built to close:** conversational like the chatbot users asked for, accurate enough to rebuild trust, and smart enough to surface the right property without forcing users into a form.

---

## Slide 3 — The Solution: A System of Specialists

- Not one model trying to do everything
- Five coordinated modules, each with one clear job, working together like a real estate office
- Built so the AI explains real data instead of generating answers from memory

| Module | Role in one line |
|---|---|
| **Agent** | Understands the user and manages the conversation |
| **Recommender** | Finds and ranks real property listings |
| **Analyzer** | Judges price fairness and market context |

---

## Slide 4 — Agent Module: The Conversation Manager

**Role:** Turns a messy, natural sentence into a clear, actionable request — and keeps track of the conversation

- Understands intent: searching, asking for market advice, booking a visit, or referring back to "the second one"
- Extracts structured details (location, budget, type, bedrooms, buy/rent) using AI *and* rule-based checks together, for reliability
- Remembers context across turns — filters, recent recommendations, which property the user is currently discussing
- Routes the request to the right module, then writes the final, natural-language reply

**Key tech:** LangGraph workflow — Understand → Use Tools → Write Reply → Audit

**Why it matters:** This is what removes the rigid search form and replaces it with a real conversation

---

## Slide 5 — Recommender Module: The Search & Ranking Engine

**Role:** Finds the best-matching real listings — not just listings that pass a filter

- Builds a structured search profile from the user's request
- Filters out anything that breaks hard rules (wrong category, over budget)
- Scores remaining properties on price fit, location, amenities, and overall value
- Ranks results and adds variety, so the user doesn't see ten nearly identical units
- Asks for missing basics (buy/rent, location, budget) instead of guessing

**Key tech:** FAISS semantic search over property descriptions + an XGBoost ranking model

**Why it matters:** Goes beyond simple filtering to actual judgment — closer to how a human agent compares options

---

## Slide 6 — Analyzer Module: The Market Intelligence Engine

**Role:** Judges whether a property and a request make sense in the real market — with evidence, not opinion

- **Fair Price Estimator** — compares a property to similar listings, gives a verdict (highly competitive / fair / premium / overpriced) with a confidence score
- **Market Pulse** — area-level stats: median price, price per m², days on market
- **Area Comparator** — compares two locations side by side
- **Preference Engine** — checks if a user's request is realistic, and suggests smart adjustments (budget, area, size)
- **Knowledge Engine** — pulls from written market documents for richer context

**Key tech:** Statistical comparison engine + retrieval over market knowledge documents (RAG)

**Why it matters:** Separates "here's a match" from "here's what the price actually means" — keeps market advice evidence-based

---

## Slide 7 — What Makes It Trustworthy

- Hybrid by design: AI handles language, real data handles facts
- AI never invents a property, a price, or a listing ID
- Every recommendation is traceable back to actual database records
- A dedicated **Auditor** step checks every response against real returned data before it reaches the user — automatically, not just as a prompt instruction

---

## Slide 8 — From Capability to Business Outcome

*(Each row reads left to right: what the system does → the user problem it removes → what that's actually worth to the business)*

| Capability | User Problem It Solves | Business Outcome |
|---|---|---|
| Auditor-grounded answers — never invents a listing | 88% of users don't fully trust online listings | Higher credibility on the platform's first impression → more browsers convert into actual inquiries |
| Recommender's ML-based ranking | 42% say it's hard to find the right property | Faster, more relevant matches → less drop-off mid-search, more saved/booked properties |
| Analyzer's fair-price verdicts | Users can't judge if a price is reasonable on their own | The platform acts as an advisor, not just a listings board — a clear differentiator |
| Arabic & English support, RTL UI | The Egyptian market communicates in both languages | No language barrier to adoption → a wider addressable user base from day one |

---

## Slide 9 — Why This Matters Strategically

- **Differentiation:** most local platforms are static listing boards; Estately offers a consultant — that's a hard feature to copy quickly
- **Trust as a growth lever:** in real estate, trust is the #1 blocker to action — solving it isn't a "nice to have" feature, it's the thing standing between a visitor and a lead
- **Scalability without linear cost:** one AI system can hold thousands of conversations at once — the equivalent of an unlimited sales team, without unlimited hiring
- **Stickier funnel:** a helpful, conversational experience keeps users engaged long enough to move from "just browsing" to booking a visit

---

## Slide 10 — Roadmap & Closing

- Recommender currently trained on simulated buyer personas
- Next step: learn from real user behavior — clicks, saves, bookings
- Estately's AI: built to be helpful *and* honest

---

## Delivery Notes

- **The problem → solution thread now runs front-to-back:** Slide 1 bridges in, Slide 2 names the gap explicitly, Slides 3–6 are the solution in detail (overview, then module by module), Slide 7 closes the trust loop, and Slides 8–9 cash it all in as business value.
- **Slide 2 is short on purpose** — it's a callback, not a fresh justification, since the team already covered this ground earlier. Land the four numbers fast and move straight into Slide 3.
- **Slide 3 is your map** — show it briefly, then say "let's walk through each one," so the audience already has the mental model before the deep dives start.
- **Slide 6 (Analyzer) and Slide 4 (Agent) tend to draw the most committee questions** — Agent because of the LangGraph workflow, Analyzer because "fair price" claims invite scrutiny. Be ready to explain the confidence score and the comparable-listings logic if asked.
- Keep slide text exactly as short as shown above — expand verbally. Committees read ahead of the speaker, so dense slide text steals attention from your talk track.
