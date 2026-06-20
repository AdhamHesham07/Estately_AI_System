## Slide 1 — Title & Bridge / The Gap That Shaped This System

**Estately AI System — The Intelligent Layer Behind the Platform**
[Your Name] — AI Engineer

*Bridge:* as you've heard, trust and discoverability were two of the biggest gaps we found — here's the system built to close them.

**The Gap:**
- **88%** don't fully trust online listings
- **42%** want an AI chatbot — the #1 requested feature
- **42%** find it hard to discover the right property
- **60%** are just exploring, not ready for a rigid search form

---

## Slide 2 — The Solution: A System of Specialists

- Not one model trying to do everything — five coordinated modules, each with one clear job
- Built so the AI explains real data, instead of generating answers from memory

| Module | Role in one line |
|---|---|
| **Agent** | Understands the user and manages the conversation |
| **Recommender** | Finds and ranks real property listings |
| **Analyzer** | Judges price fairness and market context |

---

## Slide 3 — Agent Module: The Conversation Manager

**Role:** Turns a messy, natural sentence into a clear request — and keeps track of the conversation

- Understands intent (search, market advice, booking, follow-ups) and extracts details using AI + rule-based checks together
- Remembers context across turns — filters, recent results, which property the user means
- Routes to the right module, then writes the final natural-language reply — in Arabic or English

**Trust:** A dedicated **Auditor** step checks every response against real returned data before it reaches the user, automatically — not just a prompt instruction

**Business Impact:** Delivers the AI chatbot 42% of users specifically asked for, in both languages the market actually speaks — removing the language barrier and the rigid-form drop-off

---

## Slide 4 — Recommender Module: The Search & Ranking Engine

**Role:** Finds the best-matching real listings — not just listings that pass a filter

- Builds a search profile, filters out hard-rule breaks (wrong category, over budget)
- Scores remaining properties on price fit, location, amenities, and value
- Ranks and adds variety — no ten near-identical units; asks for missing basics instead of guessing

**Key tech:** FAISS semantic search + XGBoost ranking model

**Trust:** Never recommends a property that doesn't exist in the data — every result traces back to a real database record

**Business Impact:** ML-based ranking directly answers the 42% who said finding the right property is hard — faster, more relevant matches, less drop-off, more saved/booked properties

**Roadmap:** Currently trained on simulated buyer personas; next step is learning from real user behavior — clicks, saves, bookings — as the platform gathers it

---

## Slide 5 — Analyzer Module: The Market Intelligence Engine

**Role:** Judges whether a property and a request make sense in the real market — with evidence, not opinion

- **Fair Price Estimator** — verdict (competitive / fair / premium / overpriced) with a confidence score
- **Market Pulse & Area Comparator** — median price, price/m², days on market, side-by-side areas
- **Preference Engine & Knowledge Engine** — flags unrealistic requests, pulls in written market context

**Trust:** Every verdict is backed by real comparable listings and a confidence score — not a confident-sounding sentence

**Business Impact:** Turns the platform from a listings board into an advisor — directly targets the 88% trust gap by showing *why* a price is fair, not just stating that it is

---

## Slide 6 — Why This Matters Strategically

- **Differentiation:** most local platforms are static listing boards; Estately offers a consultant — a hard feature to copy quickly
- **Trust as a growth lever:** in real estate, trust is the #1 blocker to action — this isn't a "nice to have," it's what stands between a visitor and a lead
- **Scalability without linear cost:** one AI system can hold thousands of conversations at once — the equivalent of an unlimited sales team, without unlimited hiring
- **Stickier funnel:** a helpful, conversational experience keeps users engaged long enough to move from "just browsing" to booking a visit

**Closing:** Estately's AI — built to be helpful *and* honest.

---

