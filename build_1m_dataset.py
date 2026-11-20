
from pathlib import Path
import re
import itertools
import random
import pandas as pd


TARGET_ROWS = 1_000_000
SEED = 42
random.seed(SEED)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Load the original dataset
if (DATA_DIR / "sms.tsv").exists():
    df = pd.read_csv(
        DATA_DIR / "sms.tsv",
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
        on_bad_lines="skip"
    )
elif (DATA_DIR / "spam.csv").exists():
    df = pd.read_csv(
        DATA_DIR / "spam.csv",
        encoding="latin-1"
    )
    df = df.rename(columns={"v1": "label", "v2": "message"})
    df = df[["label", "message"]]
else:
    raise FileNotFoundError(
        "Put your dataset at data/sms.tsv or data/spam.csv"
    )

df["label"] = df["label"].astype(str).str.strip().str.lower()
df["message"] = df["message"].astype(str).str.strip()

df = df[df["label"].isin(["ham", "spam"])]
df = df[df["message"].str.len() > 0]
df = df.drop_duplicates(subset=["message"]).copy()
df["source"] = "real"

# Synthetic message templates. These are examples, not real messages.
ham_templates = [
    "Hi {name}, are we still meeting at {time}?",
    "Hey {name}, just checking — are you coming at {time}?",
    "Your appointment is scheduled for {day} at {time}.",
    "Please call me when you reach {place}.",
    "I will send the {item} after the meeting.",
    "Thanks for your message. I will reply {time}.",
    "Can you bring the {item} when you come?",
    "Your class starts at {time} in {place}.",
    "I am on my way and will reach in {minutes} minutes.",
    "Please let me know if you received the {item}.",
    "Are you free to talk {time}?",
    "Don't forget the {item} for today's session.",
    "Reminder: your meeting is at {time} in {place}.",
    "Hey {name}, the {item} is ready for pickup at {place}.",
    "Just a heads-up — I'll be at {place} by {time}.",
    "Can we reschedule to {day} at {time}? Let me know.",
    "The {item} you requested will be available {time}.",
    "Hi {name}, your order is ready. Collect from {place}.",
    "Please confirm your attendance for {day}.",
    "I need the {item} by {time} if possible.",
    "Running {minutes} minutes late. See you soon!",
    "Hey, are you still at {place}? I'm nearby.",
    "Your session on {day} is confirmed for {time}.",
    "Good morning! Don't forget — {item} submission is {time}.",
    "Hi {name}, the team will meet at {place} on {day}.",
    "Can you review the {item} before {time}?",
    "Thanks {name}, I got the {item}. Will read it {time}.",
    "The bus leaves {place} at {time}. Be there!",
    "Let's catch up over lunch {day} at {place}.",
    "Just confirmed — the {item} is booked for {day}.",
    "Will be at {place} in {minutes} minutes. Wait for me!",
    "Hey {name}, did you get my last message about the {item}?",
    "Your updated {item} is ready. Pick it up from {place}.",
    "We are meeting at {place} at {time}. Don't be late!",
    "Finished the {item}. Will share it with you {time}.",
    "Can you send me the {item} before end of {day}?",
    "Hope you are well! Let's talk {time} — I have some news.",
    "Hi {name}, please review the {item} and reply by {time}.",
    "The workshop at {place} starts at {time} on {day}.",
    "Just wanted to say thanks for the {item}, {name}!",
    "Your parcel is at {place}. Collect before {time}.",
    "Reminder: {item} due {day}. Please submit on time.",
    "The team wants to meet at {time}. Is {place} okay?",
    "Hey {name}, I left the {item} at {place}. Pick it up.",
    "Call me when you can — it's about the {item}.",
    "Library closes at {time}. Make sure you return the {item}.",
    "Hi {name}, please bring the {item} to class on {day}.",
    "The {item} meeting has been moved to {time} at {place}.",
    "I'll send the {item} over by {time}. Give me a bit.",
    "Quick check — did you get the {item} from {name}?",
    "Thanks for covering the {item} today. See you {day}!",
]

spam_templates = [
    "Congratulations! You have won {prize}. Claim your reward at {url}",
    "URGENT! Your {account} needs verification. Visit {url}",
    "You are selected for a {prize}. Reply {keyword} to claim.",
    "Get a special offer of {discount}% today. Visit {url}",
    "You may have won {amount}. Call {phone} to claim now.",
    "Limited offer! Claim your {prize} before {day}. Visit {url}",
    "Your number was selected for a reward of {amount}.",
    "Earn extra cash today. Register at {url}",
    "Exclusive deal! Save {discount}% on your next purchase.",
    "Final reminder: claim your {prize} now at {url}",
    "Alert: your {account} is at risk! Secure it at {url}",
    "You've been chosen for a {prize} worth {amount}. Tap {url}",
    "Free {prize} waiting for you. Claim within {minutes} minutes at {url}",
    "Jackpot winner! You won {amount}. Verify at {url}",
    "Special {discount}% cashback offer just for you. Visit {url}",
    "Your {account} has a pending reward of {amount}. Claim at {url}",
    "Act now! {discount}% off all items. Use code {keyword} at {url}",
    "Reward alert: {amount} credited to your {account}. Confirm at {url}",
    "You qualify for a free {prize}. Limited slots — reply {keyword}.",
    "Win big! Enter now at {url} for a chance to win {prize}.",
    "Huge savings: {discount}% off today only. Shop at {url}",
    "Call {phone} now and claim your {amount} prize!",
    "Your account {account} has been selected. Get {prize} at {url}",
    "Bonus offer: get {prize} free when you register at {url}",
    "Flash sale! {discount}% off for the next {minutes} minutes at {url}",
    "Congratulations {name}! You won a {prize}. Visit {url} to claim.",
    "WINNER! Your number wins {amount}. Call {phone} immediately.",
    "Redeem your {prize} today. Reply {keyword} or visit {url}",
    "We tried reaching you about your {account} reward of {amount}.",
    "Claim your {discount}% welcome bonus now at {url}",
    "Urgent notice: {amount} transfer pending for your {account}. Visit {url}",
    "Get a free {prize} just for signing up at {url}",
    "You have {minutes} minutes to claim your {prize}. Go to {url}",
    "Secret deal: {discount}% off {prize} — use code {keyword} at {url}",
    "Your lottery entry has won {amount}! Contact us at {phone}.",
    "Special invite: earn {amount} working from home. Join at {url}",
    "Last chance! Your {prize} expires today. Click {url} now.",
    "Lucky draw winner! Collect {prize} worth {amount} at {url}",
    "New offer: {discount}% off your next {prize} purchase at {url}",
    "Your {account} reward is ready. Confirm identity at {url}",
    "Top prize of {amount} is yours. Reply {keyword} to confirm.",
    "Grab your {prize} now — only {minutes} minutes left! {url}",
    "Surprise! A {prize} is waiting for you at {url}.",
    "Earn up to {amount} daily from home. Sign up at {url}",
    "Exclusive member benefit: {discount}% off + free {prize} at {url}",
    "Verify your {account} to unlock {amount} cashback at {url}",
    "Today only: buy 1 get 1 free on {prize}. Visit {url}",
    "You are our {day} lucky winner of {prize}. Claim at {url}",
    "Your {amount} reward expires {day}. Don't miss out — {url}",
    "STOP paying full price! Get {discount}% off at {url} with code {keyword}",
]

values = {
    "name": [
        "Alex", "Sam", "Jordan", "Taylor", "Riya", "Aman", "Priya", "Rahul",
        "Neha", "Chris", "Aisha", "Daniel", "Emma", "Liam", "Sophia", "Noah",
        "Olivia", "James", "Mia", "Ethan", "Zara", "Raj", "Ananya", "Vikram",
        "Fatima", "Omar", "Yuki", "Carlos", "Maria", "Wei", "Sara", "Arjun",
        "Divya", "Karan", "Pooja", "Rohan", "Nisha", "Amit", "Sunita", "Dev",
        "Preeti", "Suresh", "Meena", "Pankaj", "Rekha", "Ajay", "Deepa", "Vinod",
        "Kavya", "Harish", "Ben", "Lily", "Max", "Zoe", "Ryan", "Chloe",
        "Lucas", "Ella", "Mason", "Ava", "Aiden", "Isla", "Jackson", "Luna",
        "Sebastian", "Grace", "Henry", "Nora", "Owen", "Layla", "Gabriel",
        "Riley", "Leo", "Zoey", "Julian", "Penelope", "Mateo", "Hannah",
        "Santiago", "Lillian", "Ezra", "Ellie", "Hudson", "Aurora", "Asher",
        "Natalie", "Elijah", "Victoria", "Grayson", "Claire", "Levi", "Elena",
        "Nathan", "Scarlett", "Caleb", "Leah", "Thomas", "Hazel",
    ],
    "time": [
        "6 PM", "tomorrow", "this evening", "10 AM", "after lunch", "in an hour",
        "at noon", "by 5 PM", "first thing in the morning", "tonight at 8",
        "this afternoon", "at 3 PM", "on Monday morning", "right after class",
        "before 9 AM", "at 7:30 AM", "around midday", "at half past two",
        "after dinner", "at 11:45 AM", "by end of day", "at 4 PM",
        "early tomorrow", "before the meeting", "as soon as possible",
        "at 8 AM", "by 2 PM", "at 9:30 AM", "late tonight", "in the morning",
        "at 1 PM", "by 6:30 PM", "at 11 AM", "this morning", "by 8 PM",
        "at dusk", "at 7 PM", "at 5:30 AM", "around 3:30 PM", "just before noon",
        "in two hours", "within the hour", "at 10:30 PM", "by midnight",
        "at first light", "after work", "before sunset", "around 4:30 PM",
        "at 2:30 PM", "by 9 PM",
    ],
    "day": [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
        "Sunday", "tomorrow", "next week", "this weekend", "the day after tomorrow",
        "next Monday", "next Friday", "this Thursday", "today",
        "next Tuesday", "next Wednesday", "next Saturday", "next Sunday",
        "this coming Monday", "this coming Friday", "in two days", "in three days",
        "next month", "at the end of the week", "by week's end", "the following day",
        "the next working day", "this coming weekend", "next long weekend",
    ],
    "place": [
        "the station", "college", "the office", "home", "the library",
        "the canteen", "the park", "the hospital", "the mall", "the lab",
        "the conference room", "the cafeteria", "Gate B", "the lobby",
        "the reception desk", "the sports hall", "the parking lot", "Block C",
        "the main entrance", "the rooftop terrace",
        "the lecture hall", "Studio 3", "the admin block", "the bus stop",
        "the gym", "the seminar room", "Floor 5", "the data centre",
        "the break room", "the waiting area", "Building A", "Room 204",
        "the print room", "the server room", "the front desk", "Gate 7",
        "the staff room", "the common room", "the rooftop", "the basement",
    ],
    "item": [
        "document", "file", "notes", "details", "report", "assignment",
        "presentation", "invoice", "proposal", "contract", "form",
        "certificate", "ticket", "package", "receipt", "brochure",
        "handbook", "schedule", "agenda", "summary",
        "spreadsheet", "draft", "brief", "memo", "flyer", "permit",
        "application", "blueprint", "checklist", "manual",
        "sample", "proof", "revision", "template", "data sheet",
        "project plan", "task list", "reference guide", "logbook", "diagram",
    ],
    "minutes": [
        "5", "10", "15", "20", "30", "45", "60", "90", "2", "3",
        "7", "12", "25", "35", "40",
    ],
    "prize": [
        "gift card", "smartphone", "cash prize", "shopping voucher",
        "luxury watch", "laptop", "tablet", "holiday package", "TV",
        "gaming console", "headphones", "fitness tracker", "camera",
        "e-wallet credit", "airline tickets", "spa voucher",
        "grocery hamper", "smart speaker", "earbuds", "power bank",
        "bicycle", "perfume set", "book bundle", "streaming subscription",
        "monthly grocery pass", "gold coin", "diamond ring", "cruise voucher",
        "hotel stay", "car service voucher",
    ],
    "url": [
        "http://claimreward.xyz/now", "https://offers.promo/win",
        "http://getprize.co/claim", "https://rewards.club/redeem",
        "http://myprize.online/collect", "https://deals.shop/special",
        "http://cashback.store/apply", "https://vouchers.pro/grab",
        "http://winnow.site/enter", "https://freegift.me/claim",
        "http://bonusoffer.cc/verify", "https://promos.buzz/now",
        "http://cashwin.io/collect", "https://myrward.net/go",
        "http://prizespot.org/confirm", "http://getbonus.win/start",
        "https://luckywin.co/claim", "http://rewardzone.biz/get",
        "https://topdeals.xyz/now", "http://specialoffer.net/grab",
        "https://mygift.io/collect", "http://winzone.club/enter",
        "https://cashprize.me/now", "http://offerlink.biz/go",
        "https://freewin.co/start", "http://bonuscash.org/claim",
        "https://promo99.net/win", "http://rewardnow.io/get",
        "https://gift4u.co/collect", "http://specialwin.xyz/now",
    ],
    "account": [
        "bank account", "mobile account", "online account",
        "savings account", "prepaid account", "digital wallet",
        "credit card account", "loyalty account", "subscription account",
        "rewards account",
    ],
    "keyword": [
        "WIN", "CLAIM", "YES", "OFFER", "GO", "GET", "FREE",
        "GRAB", "ACCEPT", "CONFIRM", "REDEEM", "OK", "JOIN", "NOW",
        "START", "ENTER", "TAKE", "UNLOCK", "ACTIVATE", "COLLECT",
        "APPLY", "RECEIVE", "VERIFY", "ACCESS", "EARN",
    ],
    "amount": [
        "500", "1000", "2000", "5000", "10000", "250", "750", "1500",
        "3000", "7500", "25000", "50000", "100", "200", "400",
        "800", "1200", "4500", "9999", "15000",
        "350", "600", "950", "1750", "2500", "4000", "6000", "8000",
        "12000", "20000",
    ],
    "phone": [
        "08000000000", "09000000000", "07000000000",
        "08001234567", "09876543210", "07712345678",
        "08099887766", "09012345678", "07600001122",
        "08055443322", "09123456789", "07811223344",
        "08077665544", "09988776655", "07533221100",
        "08044332211", "09765432109", "07699887766",
        "08022334455", "09654321098",
    ],
    "discount": [
        "10", "15", "20", "25", "30", "35", "40", "45", "50",
        "55", "60", "65", "70", "75", "80", "5", "12", "18", "22", "28",
    ],
}

def get_placeholders(template):
    """Return list of placeholder names found in a template string."""
    return re.findall(r'\{(\w+)\}', template)


def enumerate_template(template, values, label):
    """Yield every unique filled message for one template via itertools.product."""
    phs = get_placeholders(template)
    if not phs:
        yield {"label": label, "message": template, "source": "synthetic"}
        return
    opts = [values.get(p, [f"{{{p}}}"]) for p in phs]
    for combo in itertools.product(*opts):
        msg = template
        for ph, val in zip(phs, combo):
            msg = msg.replace(f"{{{ph}}}", val, 1)
        yield {"label": label, "message": msg, "source": "synthetic"}

# ── Build the dataset ──────────────────────────────────────────────────────
if len(df) >= TARGET_ROWS:
    result = df.sample(n=TARGET_ROWS, random_state=SEED).copy()
else:
    needed = TARGET_ROWS - len(df)
    print(f"Real rows: {len(df):,}  |  Synthetic needed: {needed:,}")
    print("Enumerating all template combinations ...")

    seen_real = set(df["message"].str.lower())
    synthetic_rows = []

    for label, tmpl_list in [("ham", ham_templates), ("spam", spam_templates)]:
        for tmpl in tmpl_list:
            for row in enumerate_template(tmpl, values, label):
                if row["message"].lower() not in seen_real:
                    synthetic_rows.append(row)

    print(f"Total unique synthetic messages available: {len(synthetic_rows):,}")

    if len(synthetic_rows) < needed:
        raise RuntimeError(
            f"Only {len(synthetic_rows):,} unique synthetic messages available "
            f"but {needed:,} are needed. Add more templates or values."
        )

    # Shuffle and sample exactly what we need
    rng = random.Random(SEED)
    rng.shuffle(synthetic_rows)
    chosen = synthetic_rows[:needed]

    if len(chosen) % 100_000 == 0 or True:
        print(f"Sampled {len(chosen):,} synthetic rows.")

    result = pd.concat(
        [df, pd.DataFrame(chosen)],
        ignore_index=True
    )

# Shuffle and save exactly one million rows
result = result.sample(frac=1, random_state=SEED).reset_index(drop=True)
output = DATA_DIR / "sms_spam_1million.csv"
result.to_csv(output, index=False, encoding="utf-8")

print("Saved:", output)
print("Total rows:", len(result))
print("\nLabel counts:")
print(result["label"].value_counts())
print("\nSource counts:")
print(result["source"].value_counts())
