"""Synthetic (schema, prompt) -> Lucene query training data for Needle 3.

Each example is built from a small set of structured constraints. Every constraint
renders both to natural language (many paraphrases) and to a canonical Lucene
StandardQueryParser string, so the target is correct by construction. All targets
are then checked with the real Lucene parser (lucene/Validate.java).

Canonical query style:
  * constraints joined with AND, in the order they are mentioned
  * exclusions appended as `AND NOT x`; only-negative queries use `*:* AND NOT x`
  * OR groups always parenthesised unless they are the whole query
  * multi-word or special-character values are quoted phrases
  * generic topical keywords are unfielded (searched against default fields)
  * numeric: exact `f:N`, ranges `[a TO b]`, open ends `{N TO *]`, `[* TO N}` etc.

Usage: uv run gen_data.py --n 40000 --out data/train.jsonl --seed 1
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys

from lucene_query import TOOL_NAME, tools_json

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- vocab

TOPICS_1 = """dragons wizards magic space robots vampires zombies pirates war love history
science physics chemistry biology economics politics climate energy solar wind nuclear
python java rust golang javascript kubernetes docker linux windows android ios database
security privacy encryption blockchain bitcoin finance investing stocks marketing sales
startups leadership management cooking baking vegan gluten coffee tea wine beer travel
hiking camping fishing gardening photography music jazz guitar piano poetry philosophy
psychology meditation yoga fitness running cycling soccer basketball tennis chess golf
elections democracy immigration healthcare vaccines cancer diabetes nutrition sleep
parenting education math statistics calculus algebra astronomy mars moon ocean forests
wildlife dogs cats horses birds insects dinosaurs volcanoes earthquakes hurricanes floods
drought agriculture farming manufacturing logistics shipping aviation railways cars
electric batteries hydrogen semiconductors chips quantum ai llm transformers vision
speech translation search ranking recommendation latency caching outage timeout crash
memory leak deadlock regression migration upgrade refund invoice shipping delay warranty
murder detective romance thriller horror comedy satire adventure mystery heist revenge
friendship betrayal family childhood aging grief identity freedom justice corruption
espionage samurai knights castles empires revolution renaissance medieval victorian""".split()

TOPICS_N = [
    "machine learning", "deep learning", "neural networks", "climate change", "renewable energy",
    "electric vehicles", "supply chain", "interest rates", "real estate", "stock market", "central bank",
    "artificial intelligence", "natural language processing", "computer vision", "reinforcement learning",
    "open source", "cloud computing", "data privacy", "remote work", "mental health", "public health",
    "civil war", "world war", "cold war", "space exploration", "black holes", "dark matter", "gene editing",
    "stem cells", "quantum computing", "social media", "video games", "board games", "time travel",
    "coming of age", "plant based", "gluten free", "low carb", "high protein", "customer support",
    "order status", "password reset", "payment failed", "null pointer exception", "out of memory",
    "connection refused", "disk full", "rate limit exceeded", "permission denied", "garbage collection",
    "load balancer", "message queue", "service mesh", "unit tests", "lord of the rings", "harry potter",
    "star wars", "the great gatsby", "pride and prejudice", "new york", "los angeles", "san francisco",
    "united kingdom", "middle east", "south america", "small business", "family owned", "free shipping",
    "limited edition", "best seller", "user interface", "dark mode", "battery life", "noise cancelling",
    "first edition", "short stories", "true crime", "science fiction", "magical realism", "hard boiled",
    "sourdough bread", "fried chicken", "olive oil", "credit card", "wire transfer", "late fee",
    "kernel panic", "segmentation fault", "stack overflow", "dependency injection", "blue whale",
    "solar panels", "heat pump", "wind turbine", "data center", "self driving", "facial recognition",
]
SPECIAL_TERMS = ["C++", "C#", "AT&T", "e-mail", "sci-fi", "Wi-Fi", "co-op", "x-ray", "T-shirt",
                 "node.js", "3D", "R&D", "U.S.", "Q&A", "covid-19", "k-pop", "B2B", "anti-aging",
                 "self-help", "non-fiction", "real-time", "high-res", "e-commerce", "COVID-19",
                 "follow-up", "re-entry", "user:admin", "50%", "HTTP/2", "a+b"]

FIRST = """James Mary Robert Patricia John Jennifer Michael Linda David Elizabeth William Susan
Richard Jessica Joseph Sarah Thomas Karen Daniel Nancy Matthew Lisa Anthony Betty Mark Sandra
Steven Ashley Andrew Emily Kenji Yuki Wei Mei Arjun Priya Ahmed Fatima Carlos Sofia Luca Giulia
Olga Ivan Kwame Amara Lars Ingrid Pierre Chloe Hans Greta Diego Lucia Omar Leila Raj Ananya""".split()
LAST = """Smith Johnson Williams Brown Jones Garcia Miller Davis Rodriguez Martinez Hernandez Lopez
Wilson Anderson Thomas Taylor Moore Jackson Martin Lee Thompson White Harris Clark Lewis Walker
Hall Young King Wright Scott Green Baker Adams Nelson Hill Tanaka Sato Chen Wang Li Patel Singh
Khan Ali Rossi Bianchi Petrov Ivanova Mensah Okafor Nielsen Larsen Dubois Moreau Schmidt Weber
Fernandez Silva Haddad Nakamura Kim Park Nguyen Tolkien Austen Orwell Asimov Rowling Hemingway
Tolstoy Murakami Morrison Atwood Pratchett Gaiman Herbert Kubrick Spielberg Nolan Scorsese""".split()

POOLS = {
    "genre": ["fantasy", "mystery", "romance", "thriller", "horror", "biography", "history", "poetry",
              "science fiction", "young adult", "literary fiction", "self-help", "non-fiction", "comedy",
              "drama", "documentary", "animation", "western", "noir", "memoir", "sci-fi", "crime"],
    "company": ["Penguin", "HarperCollins", "Acme", "Globex", "Initech", "Umbrella", "Stark Industries",
                "Wayne Enterprises", "Apple", "Google", "Microsoft", "Amazon", "Netflix", "Sony",
                "Samsung", "Nike", "Adidas", "Ikea", "Lego", "Tesla", "Toyota", "Random House", "O'Reilly",
                "Hachette", "Vintage", "Tor", "Orbit", "Bose", "Dell", "Lenovo", "Canon", "Nikon"],
    "city": ["London", "Paris", "Berlin", "Tokyo", "Seattle", "Boston", "Chicago", "Austin", "Denver",
             "Toronto", "Sydney", "Madrid", "Rome", "Lagos", "Nairobi", "Mumbai", "Seoul", "Lisbon",
             "New York", "San Francisco", "Los Angeles", "Mexico City", "Buenos Aires", "Cape Town",
             "Amsterdam", "Dublin", "Oslo", "Prague", "Vienna", "Portland"],
    "country": ["France", "Germany", "Japan", "Brazil", "Canada", "India", "Kenya", "Nigeria", "Italy",
                "Spain", "Mexico", "Australia", "Norway", "Egypt", "Vietnam", "Chile", "Poland",
                "United States", "United Kingdom", "South Korea", "New Zealand", "South Africa"],
    "cuisine": ["italian", "mexican", "thai", "indian", "japanese", "chinese", "french", "greek",
                "korean", "ethiopian", "vietnamese", "spanish", "lebanese", "turkish", "middle eastern"],
    "color": ["red", "blue", "green", "black", "white", "silver", "gold", "navy", "beige", "pink",
              "purple", "orange", "yellow", "gray", "dark green", "light blue", "rose gold"],
    "language": ["english", "spanish", "french", "german", "japanese", "mandarin", "portuguese",
                 "arabic", "hindi", "russian", "korean", "italian", "dutch", "swahili"],
    "level": ["ERROR", "WARN", "INFO", "DEBUG", "FATAL", "TRACE", "error", "warning", "critical"],
    "status": ["open", "closed", "pending", "resolved", "active", "archived", "draft", "published",
               "shipped", "cancelled", "in progress", "on hold", "approved", "rejected", "failed"],
    "tag": ["bug", "feature", "urgent", "documentation", "enhancement", "question", "wontfix",
            "good first issue", "performance", "security", "ui", "backend", "frontend", "regression",
            "help wanted", "breaking change", "p1", "p2", "duplicate"],
    "service": ["auth", "billing", "payments", "checkout", "search", "gateway", "api", "frontend",
                "scheduler", "worker", "notifications", "inventory", "user-service", "order-service",
                "nginx", "postgres", "redis", "kafka"],
    "category": ["electronics", "clothing", "shoes", "toys", "kitchen", "furniture", "books", "sports",
                 "beauty", "garden", "automotive", "jewelry", "office supplies", "pet supplies",
                 "home decor", "outdoor", "laptops", "headphones", "cameras", "phones"],
    "make": ["Toyota", "Honda", "Ford", "BMW", "Audi", "Tesla", "Subaru", "Mazda", "Volvo", "Kia",
             "Hyundai", "Chevrolet", "Nissan", "Porsche", "Mercedes", "Volkswagen", "Jeep"],
    "venue": ["NeurIPS", "ICML", "ACL", "CVPR", "SIGIR", "KDD", "Nature", "Science", "ICLR", "EMNLP",
              "VLDB", "OSDI", "SOSP", "arXiv", "The Lancet", "Cell"],
    "section": ["sports", "politics", "business", "technology", "opinion", "world", "science",
                "health", "arts", "travel", "weather", "local", "entertainment", "food"],
    "department": ["engineering", "sales", "marketing", "finance", "legal", "hr", "support",
                   "operations", "design", "research", "product", "customer success"],
    "skill": ["python", "java", "sql", "react", "kubernetes", "aws", "excel", "figma", "terraform",
              "spark", "go", "rust", "machine learning", "project management", "data analysis", "C++"],
    "ingredient": ["chicken", "garlic", "tomato", "basil", "tofu", "salmon", "rice", "lentils",
                   "mushrooms", "spinach", "cheese", "eggs", "chickpeas", "coconut milk", "lemon",
                   "ginger", "beef", "shrimp", "avocado", "olive oil", "peanut butter", "chocolate"],
    "os": ["linux", "windows", "macos", "android", "ios", "ubuntu", "debian", "freebsd"],
    "team": ["Lakers", "Celtics", "Yankees", "Arsenal", "Chelsea", "Barcelona", "Juventus",
             "Warriors", "Packers", "Bulls", "Liverpool", "Real Madrid"],
    "mood": ["happy", "sad", "energetic", "calm", "chill", "angry", "romantic", "dark", "upbeat"],
    "word": TOPICS_1,
}

# --------------------------------------------------------------------------- schema catalog
# text kinds: person, title, body, kw (category-like; value pool from POOLS[kind])
# numeric kinds: year, price, rating, count, duration, measure, code, percent

DOMAINS = {
    "books": (["books", "novels", "titles"], [
        (["title", "book_title", "name"], "title"), (["author", "authors", "writer"], "person"),
        (["genre", "category"], "genre"), (["summary", "description", "blurb"], "body"),
        (["publisher"], "company"), (["language"], "language"),
        (["year", "publication_year", "pub_year", "published_year"], "year"),
        (["pages", "page_count", "num_pages"], "count:pages"), (["price", "list_price"], "price"),
        (["rating", "avg_rating"], "rating"), (["isbn_count"], None)]),
    "products": (["products", "items", "listings"], [
        (["name", "product_name", "title"], "title"), (["description", "details"], "body"),
        (["brand", "manufacturer"], "company"), (["category", "department"], "category"),
        (["color", "colour"], "color"), (["price", "sale_price", "cost"], "price"),
        (["rating", "stars"], "rating"), (["stock", "quantity", "inventory"], "count:units"),
        (["weight_kg", "weight"], "measure:kg"), (["reviews", "review_count", "num_reviews"], "count:reviews")]),
    "emails": (["emails", "messages", "mail"], [
        (["subject"], "title"), (["body", "content", "text"], "body"), (["from", "sender"], "person"),
        (["to", "recipient"], "person"), (["folder", "label"], "tag"),
        (["size", "size_bytes"], "measure:bytes"), (["attachments", "attachment_count"], "count:attachments"),
        (["year", "sent_year"], "year")]),
    "logs": (["logs", "log lines", "events", "log entries"], [
        (["message", "msg", "log_message"], "body"), (["level", "severity", "log_level"], "level"),
        (["service", "app", "component"], "service"), (["host", "hostname"], "kw:host"),
        (["status", "status_code", "http_status"], "code"), (["latency_ms", "duration_ms", "response_time"], "measure:ms"),
        (["bytes", "response_bytes"], "measure:bytes"), (["user", "username", "user_id"], "person"),
        (["os", "platform"], "os")]),
    "movies": (["movies", "films"], [
        (["title", "movie_title"], "title"), (["director", "directed_by"], "person"),
        (["cast", "actors", "stars"], "person"), (["plot", "synopsis", "overview"], "body"),
        (["genre", "genres"], "genre"), (["language", "original_language"], "language"),
        (["year", "release_year"], "year"), (["runtime", "runtime_minutes", "length"], "duration:minutes"),
        (["rating", "imdb_rating", "score"], "rating"), (["budget", "box_office", "gross"], "price"),
        (["country", "production_country"], "country")]),
    "jobs": (["jobs", "job postings", "openings", "positions"], [
        (["title", "job_title", "role"], "title"), (["company", "employer"], "company"),
        (["location", "city"], "city"), (["description", "requirements"], "body"),
        (["skills", "required_skills"], "skill"), (["department", "team"], "department"),
        (["salary", "min_salary", "base_salary"], "price"),
        (["experience_years", "years_experience", "min_experience"], "count:years"),
        (["remote_pct", "remote_percent"], "percent")]),
    "homes": (["homes", "houses", "listings", "properties", "apartments"], [
        (["address", "street"], "title"), (["city", "town"], "city"), (["description", "remarks"], "body"),
        (["price", "list_price", "asking_price"], "price"), (["bedrooms", "beds"], "count:bedrooms"),
        (["bathrooms", "baths"], "count:bathrooms"), (["sqft", "square_feet", "area"], "measure:square feet"),
        (["year_built", "built"], "year"), (["agent", "listing_agent"], "person"),
        (["lot_size", "lot_acres"], "measure:acres")]),
    "papers": (["papers", "articles", "publications"], [
        (["title"], "title"), (["abstract"], "body"), (["authors", "author"], "person"),
        (["venue", "journal", "conference"], "venue"), (["keywords", "topics"], "word"),
        (["year", "pub_year"], "year"), (["citations", "citation_count", "cited_by"], "count:citations"),
        (["pages", "page_count"], "count:pages")]),
    "recipes": (["recipes", "dishes", "meals"], [
        (["name", "title", "recipe_name"], "title"), (["ingredients"], "ingredient"),
        (["instructions", "steps", "method"], "body"), (["cuisine"], "cuisine"),
        (["chef", "author"], "person"), (["prep_time", "cook_time", "total_minutes"], "duration:minutes"),
        (["calories", "kcal"], "measure:calories"), (["servings", "serves"], "count:servings"),
        (["rating"], "rating")]),
    "issues": (["issues", "tickets", "bugs", "pull requests"], [
        (["title", "summary"], "title"), (["body", "description"], "body"), (["labels", "tags"], "tag"),
        (["assignee", "owner"], "person"), (["reporter", "author", "opened_by"], "person"),
        (["state", "status"], "status"), (["comments", "comment_count"], "count:comments"),
        (["priority"], "count:priority"), (["year", "created_year"], "year"),
        (["component", "service"], "service")]),
    "news": (["news articles", "stories", "articles", "news"], [
        (["headline", "title"], "title"), (["body", "content", "text"], "body"),
        (["byline", "author", "reporter"], "person"), (["section", "desk"], "section"),
        (["word_count", "words", "length"], "count:words"), (["year", "published_year"], "year"),
        (["country", "region"], "country"), (["source", "outlet"], "company")]),
    "music": (["songs", "tracks", "albums"], [
        (["title", "track_name", "song"], "title"), (["artist", "performer"], "person"),
        (["album"], "title"), (["genre"], "genre"), (["mood"], "mood"), (["lyrics"], "body"),
        (["year", "release_year"], "year"), (["duration_sec", "length_seconds"], "duration:seconds"),
        (["bpm", "tempo"], "measure:bpm"), (["plays", "play_count", "streams"], "count:plays")]),
    "cars": (["cars", "vehicles", "used cars"], [
        (["make", "brand"], "make"), (["model"], "kw:model"), (["color", "exterior_color"], "color"),
        (["description", "notes"], "body"), (["year", "model_year"], "year"),
        (["mileage", "miles", "odometer"], "measure:miles"), (["price", "asking_price"], "price"),
        (["doors"], "count:doors"), (["seller", "dealer"], "company"), (["city", "location"], "city")]),
    "restaurants": (["restaurants", "places", "spots"], [
        (["name"], "title"), (["cuisine", "food_type"], "cuisine"), (["city", "neighborhood"], "city"),
        (["reviews_text", "review", "description"], "body"), (["rating", "stars"], "rating"),
        (["price_level", "price_tier"], "count:price_level"), (["reviews", "review_count"], "count:reviews"),
        (["owner", "chef"], "person")]),
    "people": (["people", "employees", "users", "customers", "contacts"], [
        (["name", "full_name"], "person"), (["email_domain"], "kw:domain"), (["bio", "about", "notes"], "body"),
        (["title", "job_title"], "title"), (["department", "team"], "department"), (["city", "location"], "city"),
        (["country"], "country"), (["age"], "count:years"), (["salary", "income"], "price"),
        (["tenure_years", "years_employed"], "count:years"), (["skills"], "skill")]),
    "games": (["games", "video games", "matches"], [
        (["title", "name"], "title"), (["developer", "studio"], "company"), (["genre"], "genre"),
        (["description", "summary"], "body"), (["platform"], "os"), (["year", "release_year"], "year"),
        (["price"], "price"), (["metascore", "score", "rating"], "percent"),
        (["players", "max_players"], "count:players"), (["team", "home_team"], "team")]),
}
GENERIC_NOUNS = ["documents", "results", "records", "items", "entries", "docs", "hits", "anything",
                 "things", "matches", "rows"]

KW_VALUES = {"host": ["web-01", "db02", "api-3", "prod-east-1", "worker7", "cache01", "edge-lb"],
             "model": ["Civic", "Corolla", "Model 3", "Outback", "F-150", "Camry", "Golf", "Wrangler",
                       "Mustang", "Prius", "X5", "RAV4", "Accord"],
             "domain": ["gmail.com", "example.org", "acme.io", "yahoo.com", "corp.net"]}

# --------------------------------------------------------------------------- helpers

LUCENE_SPECIAL = set('+-&|!(){}[]^"~*?:\\/ ')


def needs_quotes(v):
    return any(c in LUCENE_SPECIAL for c in v) or v in ("AND", "OR", "NOT", "TO")


def lq(v):
    """Lucene term or phrase for a value."""
    if needs_quotes(v):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return v


def fmt_num(x):
    if isinstance(x, float) and not x.is_integer():
        return f"{x:.2f}".rstrip("0").rstrip(".")
    return str(int(x))


def humanize(name):
    name = name.rsplit(".", 1)[-1]
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " ").replace(".", " ").lower()
    s = re.sub(r"\b(ms|sec|kg|pct|num|avg)\b", "", s)
    return re.sub(r"\s+", " ", s).strip() or name


def style_name(name, rng):
    r = rng.random()
    if r < 0.12 and "_" in name:
        parts = name.split("_")
        return parts[0] + "".join(p.capitalize() for p in parts[1:])
    if r < 0.16:
        return rng.choice(["doc", "meta", "attr", "f", "data"]) + "." + name
    if r < 0.19 and "_" not in name:
        return name.capitalize()
    return name


class Field:
    def __init__(self, name, kind, rng):
        self.name = name
        self.kind = kind  # e.g. "person", "title", "year", "count:pages"
        base, _, self.unit = kind.partition(":")
        self.base = base
        self.numeric = base in ("year", "price", "rating", "count", "duration", "measure", "code", "percent")
        self.type = "numeric" if self.numeric else "text"
        self.human = humanize(name)
        self.rng = rng

    def value(self, rng):
        """A text value (str) for text fields."""
        b = self.base
        if b == "person":
            r = rng.random()
            if r < 0.45:
                return rng.choice(LAST)
            if r < 0.9:
                return rng.choice(FIRST) + " " + rng.choice(LAST)
            return rng.choice(FIRST)
        if b in ("title", "body", "word"):
            return topic(rng)
        if b == "kw":
            return rng.choice(KW_VALUES[self.unit])
        return rng.choice(POOLS[b])

    def num(self, rng):
        b, u = self.base, self.unit
        if b == "year":
            return rng.randint(1950, 2025)
        if b == "price":
            if rng.random() < 0.25:
                return round(rng.choice([4.99, 9.99, 14.99, 19.99, 24.5, 49.95, 99.99, 0.99, 7.5]), 2)
            return rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 300, 500, 750,
                               1000, 1500, 2000, 5000, 10000, 50000, 100000, 250000, 500000, 1000000])
        if b == "rating":
            return rng.choice([1, 2, 3, 4, 5, 3.5, 4.5, 4.2, 4.8, 7, 8, 9, 6.5, 7.5, 8.5])
        if b == "percent":
            return rng.choice([10, 20, 25, 50, 60, 70, 75, 80, 85, 90, 95, 100, 0])
        if b == "code":
            return rng.choice([200, 201, 204, 301, 302, 400, 401, 403, 404, 429, 500, 502, 503, 504])
        if b == "duration":
            return rng.choice([5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180, 240, 300])
        if b == "count":
            if u in ("bedrooms", "bathrooms", "doors", "servings", "players", "priority", "price_level",
                     "attachments"):
                return rng.randint(0 if u == "attachments" else 1, 6)
            if u == "years":
                return rng.randint(1, 65)
            return rng.choice([1, 2, 3, 5, 10, 20, 25, 50, 100, 200, 250, 300, 400, 500, 1000, 5000])
        # measure
        return rng.choice([1, 2, 5, 10, 50, 100, 200, 250, 500, 800, 1000, 1500, 2000, 5000, 10000,
                           50000, 100000, 1024, 4096, 2.5, 0.5])


def topic(rng):
    r = rng.random()
    if r < 0.55:
        return rng.choice(TOPICS_1)
    if r < 0.95:
        return rng.choice(TOPICS_N)
    return rng.choice(SPECIAL_TERMS)


def num_text(x, field, rng):
    """Natural-language rendering of a number (value in the query stays canonical)."""
    s = fmt_num(x)
    if field.base == "year":
        return s
    if isinstance(x, int) or float(x).is_integer():
        x = int(x)
        if x >= 1000 and rng.random() < 0.3:
            s = f"{x:,}"
        elif x >= 1000 and x % 1000 == 0 and rng.random() < 0.3:
            s = f"{x // 1000}k" if x < 1000000 else (f"{x // 1000000} million" if x % 1000000 == 0 else s)
        elif x >= 1000 and x % 100 == 0 and rng.random() < 0.15:
            s = f"{x / 1000:g}k"
    if field.base == "price" and rng.random() < 0.6:
        s = "$" + s
    return s


def quote_nl(v, rng):
    r = rng.random()
    if r < 0.15:
        return f'"{v}"'
    if r < 0.22:
        return f"'{v}'"
    return v


def pick(rng, *opts):
    return rng.choice(opts)


# --------------------------------------------------------------------------- constraints
# A constraint is (nl, lucene, kind) where kind is "pos", "neg", "or" (parenthesised
# OR group) or "or2" (bare OR of AND groups, wrapped when combined with other clauses).


def c_field_term(f, rng, v=None):
    v = v if v is not None else f.value(rng)
    n, V = f.human, quote_nl(v, rng)
    b = f.base
    opts = [f"with {V} in the {n}", f"where {n} is {V}", f"{n}: {V}", f"whose {n} is {V}",
            f"with {n} {V}", f"{n} = {V}", f"{n} {V}", f"having {n} {V}", f"{n} matching {V}"]
    if b == "person":
        opts += [f"by {V}", f"by {V}", f"from {V}", f"with {n} {V}", f"where the {n} is {V}"]
        if n in ("author", "authors", "writer"):
            opts += [f"written by {V}", f"authored by {V}"]
        if n in ("director", "directed by"):
            opts += [f"directed by {V}"]
        if n in ("cast", "actors", "stars"):
            opts += [f"starring {V}", f"with {V} in the cast", f"featuring {V}"]
        if n in ("assignee", "owner"):
            opts += [f"assigned to {V}", f"owned by {V}"]
        if n in ("from", "sender"):
            opts += [f"sent by {V}", f"from {V}"]
        if n in ("to", "recipient"):
            opts += [f"sent to {V}", f"addressed to {V}"]
    elif b in ("title", "body", "word"):
        opts += [f"with {V} in the {n}", f"whose {n} contains {V}", f"{n} mentions {V}",
                 f"{n} containing {V}", f"where the {n} talks about {V}", f"with {V} in their {n}",
                 f"{n} includes {V}", f"that have {V} in the {n}"]
        if n in ("title", "name", "book title", "movie title", "headline"):
            opts += [f"titled {V}", f"called {V}", f"named {V}"]
    else:
        opts += [f"in the {V} {n}", f"with {n} {V}", f"{n} is {V}", f"where {n} = {V}",
                 f"in {n} {V}", f"{n} of {V}", f"with a {n} of {V}"]
        if b in ("city", "country"):
            opts += [f"in {V}", f"located in {V}", f"based in {V}"]
        if b == "tag":
            opts += [f"tagged {V}", f"labeled {V}", f"with the {V} label"]
        if b == "color":
            opts += [f"in {V}", f"that are {V}", f"colored {V}"]
        if b in ("company", "make"):
            opts += [f"from {V}", f"made by {V}", f"by {V}"]
        if b == "level":
            opts += [f"at {V} level", f"with level {V}", f"of severity {V}"]
        if b == "status":
            opts += [f"that are {V}", f"with status {V}", f"in {V} state"]
        if b in ("genre", "cuisine"):
            opts += [f"in the {V} {n}", f"of {n} {V}"]
    return rng.choice(opts), f"{f.name}:{lq(v)}", "pos"


def c_field_neg(f, rng):
    v = f.value(rng)
    n, V = f.human, quote_nl(v, rng)
    opts = [f"not by {V}" if f.base == "person" else f"with {n} not {V}", f"excluding {n} {V}",
            f"where {n} is not {V}", f"but not with {V} in the {n}", f"without {V} in the {n}",
            f"{n} != {V}", f"no {V} in the {n}", f"except {n} {V}", f"{n} other than {V}",
            f"whose {n} doesn't contain {V}", f"leaving out {n} {V}"]
    if f.base == "person":
        opts += [f"not written by {V}", f"excluding anything by {V}", f"not from {V}"]
    return rng.choice(opts), f"{f.name}:{lq(v)}", "neg"


def c_kw(rng):
    v = topic(rng)
    V = quote_nl(v, rng)
    opts = [f"about {V}", f"mentioning {V}", f"related to {V}", f"on {V}", f"containing {V}",
            f"that talk about {V}", f"regarding {V}", f"matching {V}", f"with {V}", f"concerning {V}",
            f"that mention {V}", f"about {V}", f"discussing {V}", f"for {V}", f"with the words {V}"]
    return rng.choice(opts), lq(v), "pos"


def c_kw_neg(rng):
    v = topic(rng)
    V = quote_nl(v, rng)
    opts = [f"not about {V}", f"without {V}", f"excluding {V}", f"that don't mention {V}",
            f"but not {V}", f"not mentioning {V}", f"minus {V}", f"no {V}", f"except ones about {V}",
            f"leaving out anything about {V}", f"that do not contain {V}", f"-{v}" if " " not in v else f"without {V}"]
    return rng.choice(opts), lq(v), "neg"


def distinct_values(f, rng, k):
    vals, tries = [], 0
    while len(vals) < k and tries < 50:
        v = f.value(rng) if f else topic(rng)
        if v not in vals:
            vals.append(v)
        tries += 1
    return vals


def nl_list(vals, rng, conj="or"):
    vals = [quote_nl(v, rng) for v in vals]
    if len(vals) == 2:
        return f"{vals[0]} {conj} {vals[1]}"
    sep = pick(rng, ", ", ", ", "; ")
    tail = pick(rng, f"{sep}{conj} ", f" {conj} ", f"{sep}")
    return sep.join(vals[:-1]) + tail + vals[-1]


def c_field_or(f, rng):
    vals = distinct_values(f, rng, rng.choice([2, 2, 2, 3, 3, 4]))
    n, L = f.human, nl_list(vals, rng, "or")
    opts = [f"where {n} is {L}", f"with {n} {L}", f"{n}: {L}", f"with {L} in the {n}",
            f"whose {n} is either {L}", f"{n} matching any of {L}", f"in {n} {L}"]
    if f.base == "person":
        opts += [f"by {L}", f"by either {L}", f"from {L}"]
    elif f.base in ("city", "country"):
        opts += [f"in {L}", f"located in {L}"]
    elif f.base in ("title", "body", "word"):
        opts += [f"whose {n} mentions {L}", f"{n} containing {L}"]
    return rng.choice(opts), f"{f.name}:(" + " OR ".join(lq(v) for v in vals) + ")", "or"


def c_field_and(f, rng):
    vals = distinct_values(f, rng, rng.choice([2, 2, 3]))
    n, L = f.human, nl_list(vals, rng, "and")
    opts = [f"whose {n} contains both {L}" if len(vals) == 2 else f"whose {n} contains all of {L}",
            f"with {L} in the {n}", f"{n} mentions {L}", f"{n} containing {L}",
            f"where the {n} has {L}", f"{n} with all of: {L}"]
    return rng.choice(opts), f"{f.name}:(" + " AND ".join(lq(v) for v in vals) + ")", "pos"


def c_kw_or(rng):
    vals = distinct_values(None, rng, rng.choice([2, 2, 3]))
    L = nl_list(vals, rng, "or")
    opts = [f"about {L}", f"mentioning {L}", f"about either {L}", f"related to {L}",
            f"that mention any of {L}", f"matching {L}", f"on {L}"]
    return rng.choice(opts), "(" + " OR ".join(lq(v) for v in vals) + ")", "or"


def c_prefix(f, rng):
    w = rng.choice([t for t in TOPICS_1 if len(t) >= 5])
    p = w[: rng.randint(3, min(5, len(w) - 1))]
    P = quote_nl(p, rng)
    if f is None:
        opts = [f"with words starting with {P}", f"with terms beginning with {P}",
                f"containing a word that starts with {P}", f"matching the prefix {P}", f"matching {p}*"]
        return rng.choice(opts), f"{p}*", "pos"
    n = f.human
    opts = [f"{n} starting with {P}", f"where {n} begins with {P}", f"with {n} prefix {P}",
            f"whose {n} starts with {P}", f"{n} beginning with {P}", f"{n} matching {p}*",
            f"with a word in the {n} starting with {P}"]
    return rng.choice(opts), f"{f.name}:{p}*", "pos"


def c_wildcard(f, rng):
    w = rng.choice([t for t in TOPICS_1 if 4 <= len(t) <= 9])
    i = rng.randrange(1, len(w) - 1)
    if rng.random() < 0.6:
        pat = w[:i] + "?" + w[i + 1:]
    else:
        pat = w[:i] + "*" + w[-1]
    P = pick(rng, pat, f'"{pat}"', f"'{pat}'")
    pre = f"{f.name}:" if f else ""
    where = f" in the {f.human}" if f else ""
    opts = [f"matching the wildcard {P}{where}", f"matching the pattern {P}{where}",
            f"with wildcard pattern {P}{where}", f"that match {P}{where}"]
    return rng.choice(opts), pre + pat, "pos"


def c_fuzzy(f, rng):
    v = rng.choice([t for t in TOPICS_1 if len(t) >= 4] + LAST)
    d = rng.choice([None, None, 1, 2])
    V = quote_nl(v, rng)
    if d is None:
        opts = [f"fuzzy matching {V}", f"similar to {V} (fuzzy)", f"approximately spelled {V}",
                f"{V} or misspellings of it", f"{V} with typos allowed", f"sounding like {V} with fuzzy search",
                f"{v}~"]
        tail = "~"
    else:
        s = "s" if d > 1 else ""
        opts = [f"{V} allowing {d} typo{s}", f"{V} within {d} edit{s}", f"{V} with edit distance {d}",
                f"fuzzy {V} with at most {d} edit{s}", f"{V} with up to {d} misspelled letter{s}",
                f"{v}~{d}"]
        tail = f"~{d}"
    nl = rng.choice(opts)
    if f is not None:
        nl = nl + pick(rng, f" in the {f.human}", f" in {f.human}", f" for the {f.human}")
        return nl, f"{f.name}:{v}{tail}", "pos"
    return nl, f"{v}{tail}", "pos"


def c_proximity(f, rng):
    a, b = rng.sample(TOPICS_1, 2)
    k = rng.choice([2, 3, 4, 5, 5, 10, 20])
    A, B = quote_nl(a, rng), quote_nl(b, rng)
    opts = [f"with {A} and {B} within {k} words of each other", f"where {A} appears near {B} (within {k} words)",
            f"{A} within {k} words of {B}", f"with {A} close to {B}, at most {k} words apart",
            f"{A} and {B} no more than {k} words apart", f'"{a} {b}"~{k}']
    nl = rng.choice(opts)
    if f is not None:
        nl = nl + pick(rng, f" in the {f.human}", f" in {f.human}")
        return nl, f'{f.name}:"{a} {b}"~{k}', "pos"
    return nl, f'"{a} {b}"~{k}', "pos"


def c_phrase(f, rng):
    v = rng.choice(TOPICS_N)
    opts = [f'with the exact phrase "{v}"', f'containing the phrase "{v}"', f'with "{v}" as a phrase',
            f'matching exactly "{v}"', f"with the exact phrase {v}"]
    nl = rng.choice(opts)
    if f is not None:
        nl = nl + pick(rng, f" in the {f.human}", f" in {f.human}")
        return nl, f'{f.name}:"{v}"', "pos"
    return nl, f'"{v}"', "pos"


REGEXES = ["[A-Z]{3}[0-9]+", "v[0-9]+\\.[0-9]+", "colou?r", "err(or)?[0-9]*", "[0-9]{4}-[0-9]{2}",
           "ab.*z", "(foo|bar)baz", "user[0-9]+", "[a-z]+ing", "inv-[0-9]{6}", "sku_[0-9]+", "h.t"]


def c_regex(f, rng):
    r = rng.choice(REGEXES)
    R = pick(rng, f"/{r}/", r, f"`{r}`")
    where = f" in the {f.human}" if f else ""
    pre = f"{f.name}:" if f else ""
    opts = [f"matching the regex {R}{where}", f"matching regular expression {R}{where}",
            f"where{where or ' the text'} matches {R}", f"that match the regexp {R}{where}"]
    return rng.choice(opts), f"{pre}/{r}/", "pos"


def c_exists(f, rng, negate=False):
    n = f.human
    if negate:
        opts = [f"with no {n}", f"missing a {n}", f"without a {n}", f"where {n} is empty",
                f"that have no {n} set", f"lacking {n}"]
        return rng.choice(opts), f"{f.name}:[* TO *]", "neg"
    opts = [f"that have a {n}", f"with a {n} set", f"where {n} exists", f"with any {n}",
            f"that have some {n}", f"where {n} is present", f"with a non-empty {n}"]
    return rng.choice(opts), f"{f.name}:[* TO *]", "pos"


def c_msm(f, rng):
    k = rng.choice([3, 3, 4, 5])
    m = rng.randint(2, k - 1)
    vals = distinct_values(f, rng, k) if f else rng.sample(TOPICS_1, k)
    L = nl_list(vals, rng, "and")
    where = f" in the {f.human}" if f else ""
    opts = [f"matching at least {m} of {L}{where}", f"with at least {m} of these: {L}{where}",
            f"containing {m} or more of {L}{where}", f"that hit at least {m} of the terms {L}{where}",
            f"mentioning at least {m} of {L}{where}"]
    pre = f"{f.name}:" if f else ""
    return rng.choice(opts), f"{pre}(" + " ".join(lq(v) for v in vals) + f")@{m}", "pos"


def c_ordered(f, rng):
    vals = rng.sample(TOPICS_1, rng.choice([2, 3, 3]))
    L = nl_list(vals, rng, "then")
    L2 = ", ".join(vals)
    where = f" in the {f.human}" if f else ""
    opts = [f"with {L2} in that order{where}", f"where {L} appear in order{where}",
            f"containing {L2} in exactly that order{where}", f"with the words {L2} appearing in sequence{where}"]
    pre = f"{f.name}:" if f else ""
    return rng.choice(opts), f"{pre}fn:ordered(" + " ".join(vals) + ")", "pos"


def c_boost_fields(f1, f2, rng):
    v = topic(rng)
    V = quote_nl(v, rng)
    k = rng.choice([2, 2, 3, 3, 5, 10, 1.5])
    K = fmt_num(k)
    opts = [f"with {V} in the {f1.human} or {f2.human}, weighting {f1.human} matches {K}x",
            f"matching {V} in {f1.human} or {f2.human} with {f1.human} boosted by {K}",
            f"{V} in either {f1.human} or {f2.human}, {f1.human} hits {K} times as important",
            f"searching {f1.human} and {f2.human} for {V}, boost {f1.human} {K}x"]
    return rng.choice(opts), f"({f1.name}:{lq(v)}^{K} OR {f2.name}:{lq(v)})", "or"


def c_boost_terms(rng):
    a, b = distinct_values(None, rng, 2)
    k = rng.choice([2, 2, 3, 4, 5, 10, 0.5])
    K = fmt_num(k)
    A, B = quote_nl(a, rng), quote_nl(b, rng)
    opts = [f"about {A} or {B}, boosting {A} by {K}", f"{A} or {B} with {A} weighted {K}x",
            f"mentioning {A} or {B}, giving {A} a boost of {K}", f"{A} (boost {K}) or {B}"]
    return rng.choice(opts), f"({lq(a)}^{K} OR {lq(b)})", "or"


def c_text_range(f, rng):
    a, b = sorted(rng.sample("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 2))
    if rng.random() < 0.3:
        a, b = a.lower(), b.lower()
    n = f.human
    opts = [f"with {n} alphabetically between {a} and {b}", f"{n} in the range {a} to {b}",
            f"whose {n} sorts between {a} and {b}", f"{n} from {a} through {b} alphabetically"]
    return rng.choice(opts), f"{f.name}:[{a} TO {b}]", "pos"


# numeric -------------------------------------------------------------------------


EXTRA = {
    ("time", "gt"): ["longer than {X}{U}", "slower than {X}{U}", "taking more than {X}{U}", "over {X}{U}"],
    ("time", "gte"): ["taking at least {X}{U}", "{X}{U} or longer", "at least {X}{U}"],
    ("time", "lt"): ["shorter than {X}{U}", "faster than {X}{U}", "under {X}{U}", "taking less than {X}{U}", "quicker than {X}{U}"],
    ("time", "lte"): ["{X}{U} or less", "taking at most {X}{U}", "within {X}{U}"],
    ("rating", "gt"): ["rated above {X}", "rated higher than {X}", "with a rating over {X}"],
    ("rating", "gte"): ["rated at least {X}", "rated {X}+", "rated {X} or better", "with {X}+ stars", "{X} stars and up"],
    ("rating", "lt"): ["rated below {X}", "rated under {X}", "rated lower than {X}"],
    ("rating", "lte"): ["rated {X} or lower", "rated at most {X}", "rated no higher than {X}"],
    ("size", "gt"): ["bigger than {X}{U}", "larger than {X}{U}", "over {X}{U}", "heavier than {X}{U}"],
    ("size", "gte"): ["at least {X}{U}", "{X}{U} or bigger", "{X}{U} or more"],
    ("size", "lt"): ["smaller than {X}{U}", "under {X}{U}", "less than {X}{U}", "lighter than {X}{U}"],
    ("size", "lte"): ["{X}{U} or smaller", "at most {X}{U}", "up to {X}{U}"],
    ("count", "gt"): ["with over {X}{U}", "having more than {X}{U}"],
    ("count", "gte"): ["with {X} or more{U}", "having at least {X}{U}"],
    ("count", "lt"): ["with under {X}{U}", "having fewer than {X}{U}"],
    ("count", "lte"): ["with no more than {X}{U}", "having at most {X}{U}"],
    ("code", "gte"): ["status {X} or higher", "with status codes {X} and above", "{X}+ status"],
    ("code", "gt"): ["status above {X}", "with a status code higher than {X}"],
    ("code", "lt"): ["status below {X}", "with status under {X}"],
}

PRESENCE_UNITS = {"attachments", "comments", "reviews", "citations", "plays", "units"}

YEARISH = {"year", "publication year", "pub year", "published year", "release year", "year built",
           "built", "model year", "sent year", "created year"}


def c_numeric(f, rng):
    n, u, b = f.human, f.unit, f.base
    presence = b == "count" and u in PRESENCE_UNITS
    op = rng.choices(["gt", "gte", "lt", "lte", "between", "eq", "betweenx", "decade", "neq", "has", "hasnot"],
                     [16, 12, 16, 12, 18, 12, 2, 4 if b == "year" else 0, 4,
                      6 if presence else 0, 4 if presence else 0])[0]
    if op == "has":
        opts = [f"with {u}", f"that have {u}", f"having {u}", f"with any {u}", f"with at least one {u.rstrip('s')}",
                f"that have some {u}", f"with one or more {u}"]
        return rng.choice(opts), f"{f.name}:{{0 TO *]", "pos"
    if op == "hasnot":
        opts = [f"with no {u}", f"without {u}", f"that have zero {u}", f"with 0 {u}", f"having no {u}"]
        return rng.choice(opts), f"{f.name}:0", "pos"
    x = f.num(rng)
    y = f.num(rng)
    while y == x:
        y = f.num(rng) if rng.random() < 0.8 else x + rng.choice([1, 5, 10, 100])
    lo, hi = min(x, y), max(x, y)
    X, LO, HI = num_text(x, f, rng), num_text(lo, f, rng), num_text(hi, f, rng)
    unit = u.replace("_", " ") if u else ""
    year = b == "year" and (n in YEARISH or rng.random() < 0.5)
    cat = "time" if (b == "duration" or u in ("ms", "seconds", "minutes")) else (
        "rating" if b in ("rating", "percent") and n in ("rating", "stars", "score", "imdb rating", "metascore")
        else ("size" if b == "measure" else b))
    abbr = {"ms": "ms", "seconds": "s", "minutes": " min", "bytes": " bytes", "kg": "kg", "miles": " miles"}.get(u, " " + u if u else "")
    price = b == "price"
    q = f.name
    if op == "gt":
        opts = [f"{n} over {X}", f"{n} greater than {X}", f"{n} above {X}", f"{n} > {X}",
                f"with {n} more than {X}", f"{n} exceeding {X}", f"{n} higher than {X}", f"{n}>{X}"]
        if unit:
            opts += [f"with more than {X} {unit}", f"over {X} {unit}", f"more than {X} {unit}"]
        if year:
            opts += [f"after {X}", f"after {X}", f"later than {X}", f"newer than {X}", f"since after {X}"]
        if price:
            opts += [f"more expensive than {X}", f"costing more than {X}", f"over {X}", f"pricier than {X}"]
        opts += [p.format(X=X, U=abbr) for p in EXTRA.get((cat, "gt"), [])]
        return rng.choice(opts), f"{q}:{{{fmt_num(x)} TO *]", "pos"
    if op == "gte":
        opts = [f"{n} of at least {X}", f"{n} >= {X}", f"{n} {X} or more", f"minimum {n} {X}",
                f"{n} no less than {X}", f"{n} at least {X}", f"{n}>={X}", f"{n} {X}+"]
        if unit:
            opts += [f"with at least {X} {unit}", f"{X}+ {unit}", f"{X} or more {unit}"]
        if year:
            opts += [f"since {X}", f"from {X} onward", f"{X} or later", f"from {X} on"]
        if price:
            opts += [f"costing at least {X}", f"{X} or more", f"priced {X} and up"]
        opts += [p.format(X=X, U=abbr) for p in EXTRA.get((cat, "gte"), [])]
        return rng.choice(opts), f"{q}:[{fmt_num(x)} TO *]", "pos"
    if op == "lt":
        opts = [f"{n} under {X}", f"{n} less than {X}", f"{n} below {X}", f"{n} < {X}",
                f"with {n} lower than {X}", f"{n}<{X}", f"{n} smaller than {X}"]
        if unit:
            opts += [f"with fewer than {X} {unit}", f"under {X} {unit}", f"less than {X} {unit}"]
        if year:
            opts += [f"before {X}", f"before {X}", f"earlier than {X}", f"prior to {X}", f"older than {X}"]
        if price:
            opts += [f"under {X}", f"cheaper than {X}", f"costing less than {X}", f"below {X}"]
        opts += [p.format(X=X, U=abbr) for p in EXTRA.get((cat, "lt"), [])]
        return rng.choice(opts), f"{q}:[* TO {fmt_num(x)}}}", "pos"
    if op == "lte":
        opts = [f"{n} of at most {X}", f"{n} <= {X}", f"{n} {X} or less", f"maximum {n} {X}",
                f"{n} no more than {X}", f"{n} up to {X}", f"{n}<={X}", f"{n} not above {X}"]
        if unit:
            opts += [f"with at most {X} {unit}", f"{X} {unit} or fewer", f"up to {X} {unit}"]
        if year:
            opts += [f"{X} or earlier", f"up to {X}", f"no later than {X}", f"through {X}"]
        if price:
            opts += [f"at most {X}", f"{X} or less", f"no more than {X}", f"costing up to {X}"]
        opts += [p.format(X=X, U=abbr) for p in EXTRA.get((cat, "lte"), [])]
        return rng.choice(opts), f"{q}:[* TO {fmt_num(x)}]", "pos"
    if op == "between":
        opts = [f"{n} between {LO} and {HI}", f"{n} from {LO} to {HI}", f"{n} in the range {LO}-{HI}",
                f"with {n} {LO} to {HI}", f"{n} {LO}..{HI}", f"{n} between {LO} and {HI} inclusive",
                f"{n} in [{LO}, {HI}]", f"{n} of {LO}-{HI}"]
        if unit:
            opts += [f"with {LO} to {HI} {unit}", f"between {LO} and {HI} {unit}"]
        if year:
            opts += [f"between {LO} and {HI}", f"from {LO} to {HI}", f"from {LO} through {HI}", f"{LO}-{HI}"]
        if price:
            opts += [f"between {LO} and {HI}", f"priced {LO}-{HI}", f"costing {LO} to {HI}"]
        return rng.choice(opts), f"{q}:[{fmt_num(lo)} TO {fmt_num(hi)}]", "pos"
    if op == "betweenx":
        opts = [f"{n} strictly between {LO} and {HI}", f"{n} between {LO} and {HI} exclusive",
                f"{n} in ({LO}, {HI})", f"{n} greater than {LO} and less than {HI}",
                f"{n} above {LO} but below {HI}"]
        return rng.choice(opts), f"{q}:{{{fmt_num(lo)} TO {fmt_num(hi)}}}", "pos"
    if op == "decade":
        d = rng.choice(range(1950, 2030, 10))
        D = pick(rng, f"{d}s", f"'{str(d)[2:]}s", f"{str(d)[2:]}s") if d < 2000 else f"{d}s"
        opts = [f"from the {D}", f"in the {D}", f"{n} in the {D}", f"from the {D} decade"]
        return rng.choice(opts), f"{q}:[{d} TO {d + 9}]", "pos"
    if op == "neq":
        opts = [f"{n} not {X}", f"{n} other than {X}", f"{n} != {X}", f"excluding {n} {X}",
                f"where {n} isn't {X}", f"except {n} {X}"]
        if year:
            opts += [f"not from {X}", f"not in {X}"]
        return rng.choice(opts), f"{q}:{fmt_num(x)}", "neg"
    # eq
    opts = [f"{n} of {X}", f"{n} = {X}", f"{n} equal to {X}", f"with {n} {X}", f"{n} is {X}",
            f"{n} exactly {X}", f"{n}:{X}", f"where {n} equals {X}"]
    if unit:
        opts += [f"with exactly {X} {unit}", f"with {X} {unit}"]
    if year:
        opts += [f"from {X}", f"in {X}", f"released in {X}", f"from the year {X}"]
    if price:
        opts += [f"costing exactly {X}", f"priced at {X}"]
    if b == "code":
        opts += [f"with status {X}", f"returning {X}", f"{X} responses" if rng.random() < 0.5 else f"that got a {X}"]
    return rng.choice(opts), f"{q}:{fmt_num(x)}", "pos"


def c_or_of_ands(text_fields, rng):
    """(f1:a AND f2:b) OR (f1:c AND f2:d), e.g. "Honda Civic or Toyota Corolla"."""
    f1, f2 = rng.sample(text_fields, 2)
    a, c = distinct_values(f1, rng, 2)
    b, d = distinct_values(f2, rng, 2)
    if rng.random() < 0.4:
        nl = pick(rng, f"either {a} {b} or {c} {d}", f"{a} {b} or {c} {d}", f"{a} {b}, or {c} {d}")
    else:
        p1, p2 = c_field_term(f1, rng, a)[0], c_field_term(f2, rng, b)[0]
        p3, p4 = c_field_term(f1, rng, c)[0], c_field_term(f2, rng, d)[0]
        nl = pick(rng, f"either {p1} and {p2}, or {p3} and {p4}", f"({p1} and {p2}) or ({p3} and {p4})",
                  f"{p1} {p2}, or {p3} {p4}")
    return nl, f"({f1.name}:{lq(a)} AND {f2.name}:{lq(b)}) OR ({f1.name}:{lq(c)} AND {f2.name}:{lq(d)})", "or2"


def c_cross_or(fields, rng):
    """(A OR B) across two different fields."""
    f1, f2 = rng.sample(fields, 2)
    parts = []
    for f in (f1, f2):
        if f.numeric:
            nl, lu, kind = c_numeric(f, rng)
            while kind == "neg":
                nl, lu, kind = c_numeric(f, rng)
        else:
            nl, lu, _ = c_field_term(f, rng)
        parts.append((nl, lu))
    nl = pick(rng, f"either {parts[0][0]} or {parts[1][0]}", f"{parts[0][0]} or {parts[1][0]}",
              f"{parts[0][0]}, or else {parts[1][0]}")
    return nl, f"({parts[0][1]} OR {parts[1][1]})", "or"


# --------------------------------------------------------------------------- examples

OPENERS = ["", "", "", "find ", "show me ", "search for ", "get ", "list ", "I want ", "I'm looking for ",
           "give me ", "look up ", "return ", "fetch ", "can you find ", "query ", "any ", "all ",
           "please find ", "pull up ", "need ", "retrieve ", "search ", "show ", "which "]
JOINS = [" and ", ", ", ", ", " and ", ", and ", "; ", " ", " that are also "]

TEXT_GENS = [("field_term", 30), ("field_neg", 8), ("kw", 14), ("kw_neg", 6), ("field_or", 7),
             ("kw_or", 3), ("field_and", 3), ("prefix", 3), ("fuzzy", 2.5), ("proximity", 2.5),
             ("phrase", 2.5), ("exists", 1.5), ("missing", 1), ("msm", 1.5), ("ordered", 1.2),
             ("regex", 1.2), ("wildcard", 1.2), ("boost_fields", 1.5), ("boost_terms", 1),
             ("text_range", 0.5), ("cross_or", 4), ("or_of_ands", 1.5)]


def gen_constraint(kind, text_fields, num_fields, all_fields, rng):
    tf = rng.choice(text_fields) if text_fields else None
    opt_f = tf if (tf and rng.random() < 0.55) else None
    if kind == "numeric":
        return c_numeric(rng.choice(num_fields), rng), num_fields
    if kind in ("field_term", "field_neg", "field_or", "field_and", "exists", "missing", "text_range") and not tf:
        kind = "kw"
    if kind == "field_term":
        return c_field_term(tf, rng), tf
    if kind == "field_neg":
        return c_field_neg(tf, rng), tf
    if kind == "kw":
        return c_kw(rng), None
    if kind == "kw_neg":
        return c_kw_neg(rng), None
    if kind == "field_or":
        return c_field_or(tf, rng), tf
    if kind == "kw_or":
        return c_kw_or(rng), None
    if kind == "field_and":
        return c_field_and(tf, rng), tf
    if kind == "prefix":
        return c_prefix(opt_f, rng), opt_f
    if kind == "fuzzy":
        return c_fuzzy(opt_f, rng), opt_f
    if kind == "proximity":
        return c_proximity(opt_f, rng), opt_f
    if kind == "phrase":
        return c_phrase(opt_f, rng), opt_f
    if kind == "exists":
        f = rng.choice(all_fields)
        return c_exists(f, rng), f
    if kind == "missing":
        f = rng.choice(all_fields)
        return c_exists(f, rng, negate=True), f
    if kind == "msm":
        return c_msm(opt_f, rng), opt_f
    if kind == "ordered":
        return c_ordered(opt_f, rng), opt_f
    if kind == "regex":
        return c_regex(opt_f, rng), opt_f
    if kind == "wildcard":
        return c_wildcard(opt_f, rng), opt_f
    if kind == "boost_fields":
        if len(text_fields) < 2:
            return c_boost_terms(rng), None
        f1, f2 = rng.sample(text_fields, 2)
        return c_boost_fields(f1, f2, rng), f1
    if kind == "boost_terms":
        return c_boost_terms(rng), None
    if kind == "text_range":
        return c_text_range(tf, rng), tf
    if kind == "or_of_ands":
        if len(text_fields) < 2:
            return c_kw_or(rng), None
        return c_or_of_ands(text_fields, rng), None
    if kind == "cross_or":
        if len(all_fields) < 2:
            return c_kw_or(rng), None
        return c_cross_or(all_fields, rng), None
    raise ValueError(kind)


def make_schema(rng):
    if rng.random() < 0.85:
        dname = rng.choice(list(DOMAINS))
        nouns, catalog = DOMAINS[dname]
        catalog = [c for c in catalog if c[1]]
        k = rng.randint(min(2, len(catalog)), len(catalog))
        chosen = rng.sample(catalog, k)
        noun_pool = nouns + GENERIC_NOUNS[:4]
    else:  # mix fields from several domains
        pool = [c for d in DOMAINS.values() for c in d[1] if c[1]]
        chosen = rng.sample(pool, rng.randint(2, 9))
        noun_pool = GENERIC_NOUNS
    fields, names = [], set()
    for aliases, kind in chosen:
        name = style_name(rng.choice(aliases), rng)
        if name.lower() in names:
            continue
        names.add(name.lower())
        fields.append(Field(name, kind, rng))
    rng.shuffle(fields)
    return fields, noun_pool


PREMOD_BASES = {"genre", "color", "level", "make", "company", "cuisine", "category", "language",
                "section", "department", "person", "mood", "status", "tag", "service", "os"}


def make_example(rng):
    fields, nouns = make_schema(rng)
    schema = {f.name: f.type for f in fields}
    text_fields = [f for f in fields if not f.numeric]
    num_fields = [f for f in fields if f.numeric]

    if rng.random() < 0.006:
        nl = rng.choice(["everything", "all documents", "show me everything", "match all",
                         f"all {rng.choice(nouns)}", f"list every {rng.choice(nouns).rstrip('s')}"])
        return schema, nl, "*:*"

    n = rng.choices([0, 1, 2, 3, 4, 5], [3, 24, 34, 26, 12, 4])[0]
    gens = TEXT_GENS + ([("numeric", 38)] if num_fields else [])
    names, weights = zip(*gens)
    parts, used = [], set()
    tries = 0
    while len(parts) < n and tries < 30:
        tries += 1
        kind = rng.choices(names, weights)[0]
        (nl, lu, pol), f = gen_constraint(kind, text_fields, num_fields, fields, rng)
        key = getattr(f, "name", None) if not isinstance(f, list) else None
        if kind == "numeric":
            key = lu.split(":", 1)[0]
        if key and key in used:
            continue
        if key:
            used.add(key)
        if lu in [p[1] for p in parts]:
            continue
        parts.append((nl, lu, pol))

    # lowercase prompt sometimes (values change too, keep consistent)
    lower = rng.random() < 0.2

    noun = rng.choice(nouns)
    opener = rng.choice(OPENERS)
    premods = [f for f in text_fields if f.base in PREMOD_BASES and f.name not in used]
    premod = None
    if premods and (not parts or rng.random() < 0.3):
        f = rng.choice(premods)
        v = f.value(rng)
        premod = (f"{v}'s" if f.base == "person" and rng.random() < 0.3 else v, f"{f.name}:{lq(v)}")
    elif not parts:
        return make_example(rng)

    pos = ([(None, premod[1], "pos")] if premod else []) + [p for p in parts if p[2] != "neg"]
    neg = [p for p in parts if p[2] == "neg"]
    if pos:
        if len(pos) == 1 and not neg and pos[0][2] == "or" and pos[0][1].startswith("("):
            q = pos[0][1][1:-1]
        elif len(pos) == 1 and not neg:
            q = pos[0][1]
        else:
            q = " AND ".join("(" + p[1] + ")" if p[2] == "or2" else p[1] for p in pos)
    else:
        q = "*:*"
    for p in neg:
        q += " AND NOT " + p[1]

    if premod:
        nl = f"{opener}{premod[0]} {noun}" + ("" if not parts else " " + parts[0][0])
        for p in parts[1:]:
            nl += rng.choice(JOINS) + p[0]
        if lower:
            nl, q = nl.lower(), lower_values(q)
        if rng.random() < 0.25:
            nl = nl[:1].upper() + nl[1:]
        return schema, nl.strip(), q
    first = parts[0][0]
    rest = [p[0] for p in parts[1:]]
    if rng.random() < 0.12 and len(parts) > 1:
        body = ", ".join(p[0] for p in parts)  # terse list style without noun
        nl = opener + body
    else:
        nl = f"{opener}{noun} {first}"
        for r in rest:
            nl += rng.choice(JOINS) + r
    if lower:
        nl = nl.lower()
        q = lower_values(q)
    r = rng.random()
    if r < 0.25:
        nl = nl[:1].upper() + nl[1:]
    if r < 0.1:
        nl += "?" if opener.strip() in ("can you find", "which") else "."
    return schema, nl.strip(), q


def lower_values(q):
    """Lowercase values but not field names, operators, or regex/escape structure."""
    out, i = [], 0
    for tok in re.split(r'(\s+|"[^"]*"|/[^/]*/|\(|\)|\[|\]|\{|\})', q):
        if not tok:
            continue
        if tok in ("AND", "OR", "NOT", "TO") or tok.startswith("/") or tok.startswith("fn:"):
            out.append(tok)
        elif ":" in tok and not tok.startswith('"'):
            fld, _, val = tok.partition(":")
            out.append(fld + ":" + val.lower())
        else:
            out.append(tok.lower())
    return "".join(out)


# The cactus engine has the model write a <think> block before every tool call, so
# training examples need one too (rendered from "reasoning"); without it the tuned
# model is decoded in a context it never saw and accuracy drops sharply (77% -> 50%).
REASONING = "search tool selected."

# --------------------------------------------------------------------------- validation


def validate(rows):
    """Run each target through Lucene's StandardQueryParser. Returns list of (ok, info)."""
    lines = []
    for schema, _, q in rows:
        nums = ",".join(k for k, t in schema.items() if t == "numeric")
        lines.append(nums + "\t" + q)
    proc = subprocess.run(["java", "-cp", os.path.join(HERE, "lucene", "lib", "*"),
                           os.path.join(HERE, "lucene", "Validate.java")],
                          input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
    out = proc.stdout.rstrip("\n").split("\n")
    assert len(out) == len(rows), (len(out), len(rows), proc.stderr[-2000:])
    return [(o.startswith("OK"), o.split("\t", 1)[1]) for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--out", default="data/train.jsonl")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--show", type=int, default=0, help="print N samples and exit")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if args.show:
        for _ in range(args.show):
            s, nl, q = make_example(rng)
            print(json.dumps(s), "\n  ", nl, "\n  =>", q)
        return

    rows, seen = [], set()
    while len(rows) < args.n:
        s, nl, q = make_example(rng)
        key = (nl, json.dumps(s))
        if key in seen:
            continue
        seen.add(key)
        rows.append((s, nl, q))
    results = validate(rows)
    bad = [(r, info) for r, (ok, info) in zip(rows, results) if not ok]
    for (s, nl, q), info in bad[:20]:
        print("INVALID:", q, "|", info, file=sys.stderr)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        for (s, nl, q), (ok, _) in zip(rows, results):
            if not ok:
                continue
            fh.write(json.dumps({"query": nl, "tools": json.loads(tools_json(s)),
                                 "reasoning": REASONING,
                                 "answers": [{"name": TOOL_NAME, "arguments": {"query": q}}],
                                 "schema": s}, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows) - len(bad)} examples to {args.out} ({len(bad)} invalid dropped)")


if __name__ == "__main__":
    main()
