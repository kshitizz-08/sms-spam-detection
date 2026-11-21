"""Module 2: Advanced Multilingual NLP Text Preprocessing Pipeline.
Performs text normalization, Indic concept translation (Hindi/Marathi/Hinglish),
regex entity replacement, punctuation removal, stopword filtering,
and sentence-level spam highlighting.
100% offline — zero external APIs or downloads required.
"""
import re

# ── English Stopwords ────────────────────────────────────────────
STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any","are",
    "aren't","as","at","be","because","been","before","being","below","between","both",
    "but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't",
    "doing","don't","down","during","each","few","for","from","further","had","hadn't",
    "has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here",
    "here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll",
    "i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's",
    "me","more","most","mustn't","my","myself","no","nor","not","of","off","on",
    "once","only","or","other","ought","our","ours","ourselves","out","over","own",
    "same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some",
    "such","than","that","that's","the","their","theirs","them","themselves","then",
    "there","there's","these","they","they'd","they'll","they're","they've","this",
    "those","through","to","too","under","until","up","very","was","wasn't","we",
    "we'd","we'll","we're","we've","were","weren't","what","what's","when","when's",
    "where","where's","which","while","who","who's","whom","why","why's","with",
    "won't","would","wouldn't","you","you'd","you'll","you're","you've","your",
    "yours","yourself","yourselves"
}

# ── Devanagari numerals to ASCII digits ──────────────────────────
DEVA_NUMS = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

# ══════════════════════════════════════════════════════════════════
#  INDIC CONCEPT MAPPING (HINDI & MARATHI TO SEMANTIC ENGLISH TOKENS)
#  Allows 5 local ML models to classify Indic spam without external APIs
# ══════════════════════════════════════════════════════════════════
INDIC_SEMANTIC_MAP = {
    # ── Prize / Lottery / Rewards ──
    'अभिनंदन': 'congratulations winner prize claim',
    'बधाई': 'congratulations winner prize claim',
    'शुभकामनाएं': 'congratulations winner prize',
    'जीता': 'won winner prize claim cash',
    'जीत': 'won winner prize claim cash',
    'जिंकला': 'won winner prize claim cash',
    'जिंकलात': 'won winner prize claim cash',
    'जिंकले': 'won winner prize claim cash',
    'इनाम': 'prize reward cash claim',
    'बक्षीस': 'prize reward cash claim',
    'पारितोषिक': 'prize reward cash claim',
    'लॉटरी': 'lottery prize winner cash claim',
    'लकी': 'lucky winner prize draw',
    'ड्रॉ': 'draw prize winner lottery',
    'विजेता': 'winner prize claim reward',
    'उपहार': 'gift prize voucher free reward',
    'गिफ्ट': 'gift prize voucher free reward',
    'कूपन': 'coupon voucher discount gift',
    'वाउचर': 'voucher coupon discount gift',
    'कॅशबॅक': 'cashback money cash prize reward',
    'कॅश': 'cash money prize',
    'कॅशबॅग': 'cashback money prize cash',
    'रिवॉर्ड': 'reward prize bonus cash',
    'बोनस': 'bonus prize reward cash',

    # ── Urgency / Action triggers ──
    'तातडीने': 'urgent immediately now call action',
    'त्वरित': 'urgent immediately now call action',
    'लवकर': 'urgent fast quickly now',
    'आत्ताच': 'immediately right now today urgent',
    'तुरंत': 'urgent immediately now call action',
    'तत्काल': 'urgent immediately now priority',
    'जल्दी': 'urgent fast quickly now',
    'अत्यावश्यक': 'urgent critical immediately',
    'आजच': 'today now immediately urgent',
    'आज': 'today now urgent',
    'झटपट': 'urgent instant fast quick loan',
    'मिनिटात': 'minutes quick instant urgent',
    'मिनट': 'minutes quick instant urgent',
    'शेवटची संधी': 'urgent last chance expire limited',
    'अंतिम तारीख': 'urgent deadline expire last chance',
    'मुदत संपेल': 'expire urgent deadline last day',

    # ── Loan / Credit / Finance ──
    'कर्ज': 'loan credit instant money urgent cash offer claim apply call',
    'कर्जा': 'loan credit instant money urgent cash offer claim apply call',
    'लोन': 'loan credit instant money urgent cash offer claim apply call',
    'पाहिजे': 'offer available apply loan need',
    'पर्सनल लोन': 'personal loan instant credit money urgent offer',
    'कागदपत्रांशिवाय': 'no documents instant loan easy approval',
    'दस्तावेज़': 'documents paper loan proof',
    'मंजूर': 'approved preapproved loan credit',
    'मंजूरी': 'approval sanctioned loan credit',
    'स्वीकृत': 'approved sanctioned credit loan',
    'व्याजदर': 'interest rate low loan offer',
    'व्याज': 'interest loan payment discount',
    'हप्ता': 'emi installment loan payment',

    # ── Bank / Account / Phishing / OTP ──
    'बँक': 'bank account verify security update',
    'बैंक': 'bank account verify security update',
    'खाते': 'bank account blocked verify update',
    'खाता': 'bank account blocked verify update',
    'खात्यात': 'account bank transfer deposit money',
    'बंद': 'blocked locked suspended account urgent',
    'अवरोधित': 'blocked suspended account locked',
    'निलंबित': 'suspended blocked account locked',
    'ओटीपी': 'otp code secret password verify do not share',
    'पासवर्ड': 'password security pin code verify',
    'पिन': 'pin code password bank security',
    'एटीएम': 'atm card bank debit pin',
    'डेबिट': 'debit card bank account pin',
    'क्रेडिट': 'credit card limit bank reward',
    'केवायसी': 'kyc verification update link urgent account',
    'अपडेट': 'update verify account kyc link',
    'सत्यापित': 'verify verification account security link',
    'सत्यापन': 'verify verification account security link',
    'व्हेरिफाय': 'verify verification link account',

    # ── Money / Amounts ──
    'पैसे': 'money cash credit payment',
    'रुपये': 'money rupees cash credit',
    'रुपया': 'money rupees cash credit',
    'लाख': 'thousand money cash prize amount',
    'करोड़': 'million money cash prize amount',
    'करोड': 'million money cash prize amount',
    'हजार': 'thousand money cash rupees',
    'रक्कम': 'amount cash money sum balance',
    'धनराशि': 'amount cash money fund balance',
    'खाते शिल्लक': 'account balance money funds',

    # ── Free / Offers / Discount ──
    'मोफत': 'free offer gift discount claim zero',
    'मुफ्त': 'free offer gift discount claim zero',
    'विनामूल्य': 'free complimentary offer claim zero',
    'निःशुल्क': 'free complimentary offer claim zero',
    'फ्री': 'free offer gift claim zero',
    'ऑफर': 'offer free discount deal limited',
    'सवलत': 'discount offer deal save cash',
    'छूट': 'discount offer deal save cash',
    'बंपर': 'bumper prize lottery offer reward',
    'धमाका': 'exclusive super deal offer prize',

    # ── Links / Calls / Action requests ──
    'क्लिक': 'click link here now website urltoken',
    'लिंक': 'link website urltoken click here',
    'खालील': 'following link below click',
    'खाली': 'below link click here',
    'कॉल': 'call telephone contact number urgent',
    'संपर्क': 'contact call reach telephone',
    'डाउनलोड': 'download app install link click',
    'अर्ज': 'apply register claim submit now',
    'मिळवा': 'claim receive get collect win cash',
    'पाएं': 'claim receive get collect win cash',
    'दावा': 'claim get prize collect now',
    'तपासा': 'check verify click link now',

    # ── Safe Conversational Words (Prevents False Positives) ──
    'नमस्कार': 'hello friend greeting',
    'नमस्ते': 'hello friend greeting',
    'हाय': 'hello hi friend',
    'मित्र': 'friend buddy',
    'दोस्त': 'friend buddy',
    'आई': 'mother family',
    'बाबा': 'father family',
    'भाऊ': 'brother family',
    'ताई': 'sister family',
    'अभ्यास': 'study homework class exam assignment',
    'पढ़ाई': 'study homework class exam assignment',
    'कॉलेज': 'college university campus class study',
    'शाळा': 'school class student study homework',
    'परीक्षेची': 'exam preparation study student class',
    'परीक्षा': 'exam test student class study',
    'भेटू': 'meet coffee dinner lunch tomorrow evening',
    'भेटा': 'meet coffee dinner lunch tomorrow evening',
    'संध्याकाळी': 'evening coffee dinner meet',
    'सकाळी': 'morning breakfast coffee meet',
    'उद्या': 'tomorrow afternoon meet lunch',
    'काल': 'yesterday met discussed',
    'घर': 'home reach family safe',
    'जेवण': 'lunch dinner food eating',
    'प्रकल्प': 'project assignment code study presentation',
    'असाइनमेंट': 'assignment homework project code presentation',
}

# Hinglish (Roman-script Hindi) keywords
HINGLISH_SPAM = {
    "jeeta": "won prize claim",
    "jeet": "won prize claim",
    "inaam": "prize reward cash",
    "inam": "prize reward cash",
    "lotri": "lottery prize winner",
    "lottery": "lottery prize winner",
    "muft": "free gift claim",
    "mufat": "free gift claim",
    "mufte": "free gift claim",
    "jaldi": "urgent fast now",
    "abhi": "urgent immediately now",
    "turant": "urgent immediately call",
    "tatkal": "urgent immediately now",
    "otp": "otp code password verify",
    "atm": "atm card bank pin",
    "pin": "pin password bank verify",
    "bank": "bank account verify",
    "khata": "bank account locked update",
    "khate": "bank account locked update",
    "karj": "loan credit urgent cash",
    "kamai": "earn income daily cash",
    "kamao": "earn income daily cash",
    "badhai": "congratulations winner prize",
}

# Regex helpers
_URL_RE   = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
_PHONE_RE = re.compile(r'\b(?:\+91[-\s]?)?\d{7,}\b')
_MONEY_RE = re.compile(r'[$£€₹]\s*[\d,]+|[\d,]+\s*(?:rs\.?|inr|usd|gbp)', re.IGNORECASE)
_DEVA_RE  = re.compile(r'[\u0900-\u097F]')   # Devanagari Unicode range


def detect_script(text: str) -> str:
    """
    Detects the script / language of the SMS.
    Returns: 'hindi' | 'marathi' | 'hinglish' | 'english'
    """
    text_str = str(text or "")
    if _DEVA_RE.search(text_str):
        # Look for Marathi-exclusive words
        marathi_markers = {'आहे', 'नाही', 'आणि', 'करा', 'मिळवा', 'बक्षीस', 'खाते', 'तातडीने', 'झाले', 'द्या'}
        words = set(re.sub(r'[।,!?।.]', ' ', text_str).split())
        if words & marathi_markers:
            return "marathi"
        return "hindi"

    words_lower = set(text_str.lower().split())
    if words_lower & set(HINGLISH_SPAM.keys()):
        return "hinglish"
    return "english"


def clean_text(text: str) -> str:
    """
    Standard clean text for machine learning pipelines.
    Supports English, Devanagari (Hindi/Marathi), and Hinglish.
    Transforms Indic semantic concepts into normalized tokens so local
    pre-trained models can accurately classify them without any external API.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # 1. Normalize Devanagari numerals to standard digits (०-९ -> 0-9)
    for deva_digit, ascii_digit in DEVA_NUMS.items():
        text = text.replace(deva_digit, ascii_digit)

    # 2. Lowercase English portions
    text = text.lower()

    # 3. Delivery / Parcel Phishing Detection & Semantic Normalization
    delivery_terms = ('package', 'parcel', 'delivery', 'shipment', 'courier', 'dispatch')
    delivery_hooks = (
        'cannot be delivered', 'unable to deliver', 'failed delivery', 'on hold',
        'incomplete address', 'missing address', 'customs fee', 'delivery fee',
        'handling fee', 'pay', 'reschedule', 'pending delivery', 'track-parcel',
        'action required', 'unpaid fee', 'redelivery', 'fee'
    )
    has_delivery = any(d in text for d in delivery_terms)
    has_hook = any(h in text for h in delivery_hooks)
    has_url = bool(re.search(r'https?://|www\.|\.com|\.link|\.xyz|\.top|\.org|\.me|\.info|\.site|\.co', text))

    is_delivery_phish = has_delivery and (has_hook or has_url)

    # 4. Normalize URLs (preserving 'http' for TF-IDF feature vocabulary)
    text = re.sub(r"https?://\S+|www\.\S+", " http urltoken ", text)

    # 5. Normalize email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " emailtoken ", text)

    # 6. Normalize currency symbols ($ £ € ₹)
    text = re.sub(r"[\$£€₹]", " moneytoken ", text)

    # 7. Normalize phone numbers / long codes
    text = re.sub(r"\b\d{5,12}\b", " numtoken ", text)

    # 8. Append strong phishing tokens if delivery scam pattern matched
    if is_delivery_phish:
        text += " urgent claim verify fee prize link http code "

    # 7. Semantic concept translation for Hindi and Marathi Devanagari words
    if _DEVA_RE.search(text):
        for indic_kw, eng_equiv in INDIC_SEMANTIC_MAP.items():
            if indic_kw in text:
                text = text.replace(indic_kw, f" {eng_equiv} ")

    # 8. Hinglish concept mapping
    words = text.split()
    translated_words = []
    for w in words:
        clean_w = re.sub(r"[^\w]", "", w)
        if clean_w in HINGLISH_SPAM:
            translated_words.append(HINGLISH_SPAM[clean_w])
        else:
            translated_words.append(w)
    text = " ".join(translated_words)

    # 9. Remove non-ASCII punctuation & special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # 10. Normalize whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_with_steps(text: str) -> dict:
    """Returns a step-by-step breakdown of the preprocessing pipeline for demo/viva."""
    raw = str(text or "")
    s1 = raw.lower()
    for d, a in DEVA_NUMS.items():
        s1 = s1.replace(d, a)

    s2 = re.sub(r"https?://\S+|www\.\S+", " urltoken ", s1)
    s2 = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " emailtoken ", s2)
    s2 = re.sub(r"[\$£€₹]", " moneytoken ", s2)
    s2 = re.sub(r"\b\d{5,12}\b", " numtoken ", s2)

    s3 = clean_text(raw)
    tokens = s3.split()
    filtered = [w for w in tokens if w not in STOPWORDS]

    return {
        "original": raw,
        "step1_lowercased": s1,
        "step2_entities_replaced": s2.strip(),
        "step3_punct_removed": s3,
        "step4_stopwords_removed": " ".join(filtered),
        "token_count": len(tokens),
        "clean_token_count": len(filtered),
        "tokens": tokens,
        "clean_tokens": filtered,
    }


def check_hindi_marathi_spam(text: str) -> dict:
    """
    Analyses a Hindi/Marathi SMS for spam patterns using offline dictionary matching.
    Returns a dict with verdict, score, and detected indicators.
    """
    text_str = str(text or "")
    found_kws = {}
    for kw, equiv in INDIC_SEMANTIC_MAP.items():
        if kw in text_str and any(cat in equiv for cat in ['loan', 'prize', 'urgent', 'otp', 'free', 'click']):
            found_kws[kw] = equiv.split()[0]

    urls = _URL_RE.findall(text_str)
    phones = _PHONE_RE.findall(text_str)
    money = _MONEY_RE.findall(text_str)

    score = (
        len(found_kws) * 1.5 +
        len(urls) * 3.0 +
        len(phones) * 2.0 +
        len(money) * 1.5
    )
    score_norm = min(score / 6.0, 1.0)
    is_spam = score_norm >= 0.25 or bool(urls) or bool(phones)

    return {
        "is_spam": is_spam,
        "spam_score": score_norm,
        "keywords_found": found_kws,
        "urls_found": urls,
        "phones_found": phones,
        "money_found": money,
    }

_STYLES = {
    "url": "background:#FEE2E2;color:#991B1B;border-radius:4px;padding:2px 6px;font-weight:700;",
    "phone": "background:#FEF3C7;color:#92400E;border-radius:4px;padding:2px 6px;font-weight:700;",
    "money": "background:#FEF3C7;color:#92400E;border-radius:4px;padding:2px 6px;font-weight:700;",
    "spam": "background:#FECACA;color:#7F1D1D;border-radius:4px;padding:2px 6px;font-weight:700;",
    "devanagari_spam": "background:#FECACA;color:#7F1D1D;border-radius:4px;padding:2px 6px;font-weight:700;",
}

def highlight_message_html(text: str, spam_vocab_set: set) -> str:
    """
    Returns an HTML string of `text` with scam indicators highlighted:
      🔴 Red   — spam keywords (English / Hindi / Marathi)
      🟠 Red   — URLs
      🟡 Yellow — phone numbers and money amounts
    """
    words = text.split()
    result = []

    for word in words:
        word_bare = re.sub(r'[^\w₹$£€]', '', word)

        # 1. URL
        if _URL_RE.search(word):
            result.append(f'<span style="{_STYLES["url"]}" title="Suspicious Link">🔗 {word}</span>')
        # 2. Phone
        elif _PHONE_RE.search(word_bare) and len(re.sub(r'\D', '', word_bare)) >= 7:
            result.append(f'<span style="{_STYLES["phone"]}" title="Phone Number">📱 {word}</span>')
        # 3. Currency / Money
        elif _MONEY_RE.search(word) or any(c in word for c in ['₹', '$', '£', '€']):
            result.append(f'<span style="{_STYLES["money"]}" title="Money Amount">💰 {word}</span>')
        # 4. Devanagari spam concept
        elif _DEVA_RE.search(word):
            deva_bare = re.sub(r'[।,!?।.]', '', word)
            matched = any(kw in deva_bare for kw in INDIC_SEMANTIC_MAP if 'loan' in INDIC_SEMANTIC_MAP[kw] or 'prize' in INDIC_SEMANTIC_MAP[kw] or 'urgent' in INDIC_SEMANTIC_MAP[kw] or 'otp' in INDIC_SEMANTIC_MAP[kw] or 'free' in INDIC_SEMANTIC_MAP[kw] or 'click' in INDIC_SEMANTIC_MAP[kw])
            if matched:
                result.append(f'<span style="{_STYLES["devanagari_spam"]}" title="Scam Keyword">🚨 {word}</span>')
            else:
                result.append(word)
        # 5. English spam triggers & vocab words
        else:
            w_lower = word_bare.lower()
            cleaned = clean_text(word_bare)
            if w_lower in {'pay', 'delivered', 'locked', 'claim', 'winner', 'urgent', 'verify', 'reward', 'prize', 'lottery'}:
                result.append(f'<span style="{_STYLES["spam"]}" title="Scam Trigger">🚨 {word}</span>')
            elif cleaned and any(part in spam_vocab_set for part in cleaned.split()):
                result.append(f'<span style="{_STYLES["spam"]}" title="Spam Keyword">🚨 {word}</span>')
            else:
                result.append(word)

    return ' '.join(result)
