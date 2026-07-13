"""
ML-based Priority Classifier for Voice Support Tickets.

Architecture:
  - RandomForestClassifier trained on 28-feature vectors
  - Features: VADER sentiment scores, binary sentiment flags, urgent keyword presence/count,
    category one-hot encoding, category × sentiment interaction terms, surface text stats
  - Post-model layer: Category-based boost rules that can upgrade (never downgrade) ML predictions

Category-based boost rules (override to High):
  - Account Access  + Negative sentiment (compound ≤ -0.05)  → High  (security-sensitive)
  - Billing Issue   + Negative sentiment (compound ≤ -0.10)  → High
  - Refund Request  + Negative sentiment (compound ≤ -0.20)  → High
  - Technical Issue + Very negative     (compound ≤ -0.40)  → High
  - ANY category    + urgent keyword present                  → High (always)
"""

import os
import pickle
import ssl
import numpy as np
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('vader_lexicon', quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ─── Constants ────────────────────────────────────────────────────────────────

URGENT_KEYWORDS = [
    "compromised", "hacked", "stolen", "unauthorized", "legal", "lawyer",
    "fraud", "chargeback", "police", "refund immediately", "asap", "broken",
    "emergency", "urgent", "crash", "down", "not working", "cannot access",
    "security breach", "stolen card", "identity theft", "lawsuit", "sue",
    "cancel immediately", "unauthorized charge", "leaked", "breached", "compromise"
]

NEGATIVE_INTENSIFIERS = [
    'cannot', "can't", 'unable', 'failed', 'terrible', 'horrible',
    'awful', 'worst', 'disgusting', 'furious', 'angry', 'frustrated',
    'unacceptable', 'ridiculous', 'outrageous', 'appalling', 'nightmare'
]

CATEGORIES = [
    "Account Access", "Billing Issue", "General Inquiry",
    "Refund Request", "Technical Issue"
]

# Category → compound threshold: if compound ≤ threshold, force High priority
CATEGORY_BOOST_RULES = {
    "Account Access": -0.05,   # Security-sensitive: any negativity → High
    "Billing Issue":  -0.10,   # Financial: moderate negativity → High
    "Refund Request": -0.20,   # Refund frustration → High
    "Technical Issue": -0.40,  # Only severely negative tech issues → High
    # "General Inquiry": no boost (handled by general rules)
}


# ─── Classifier ───────────────────────────────────────────────────────────────

class PriorityClassifier:
    def __init__(self):
        self._sia = None
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(["High", "Medium", "Low"])
        self.is_trained = False
        self.train_metrics = {}

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_sia(self):
        if self._sia is None:
            self._sia = SentimentIntensityAnalyzer()
        return self._sia

    def extract_features(self, text, category):
        """
        Build a 28-dimensional feature vector for a (text, category) pair.

        Features (in order):
          [0-3]   VADER: compound, pos, neg, neu
          [4-6]   Sentiment flags: is_positive, is_negative, is_neutral
          [7-8]   Keyword: has_urgent (0/1), urgent_count (normalised 0-1)
          [9-12]  Surface: text_length, exclamation, question, neg_intensifier_count
          [13-17] Category one-hot (5 categories)
          [18-22] Category × is_negative interaction (5 features)
          [23-27] Category × has_urgent interaction (5 features)
        """
        sia = self._get_sia()
        text_str = str(text)
        scores = sia.polarity_scores(text_str)
        compound = scores['compound']

        is_positive = 1 if compound >= 0.05 else 0
        is_negative = 1 if compound <= -0.05 else 0
        is_neutral  = 1 - is_positive - is_negative

        text_lower = text_str.lower()
        urgent_matches = [kw for kw in URGENT_KEYWORDS if kw in text_lower]
        has_urgent    = 1 if urgent_matches else 0
        urgent_count  = min(len(urgent_matches) / 5.0, 1.0)

        text_length   = min(len(text_str) / 500.0, 1.0)
        exclamation   = min(text_str.count('!') / 3.0, 1.0)
        question      = min(text_str.count('?') / 3.0, 1.0)
        neg_int_count = min(
            sum(1 for w in NEGATIVE_INTENSIFIERS if w in text_lower) / 5.0, 1.0
        )

        cat_vec              = [1 if category == c else 0 for c in CATEGORIES]
        cat_neg_interaction  = [(1 if category == c else 0) * is_negative  for c in CATEGORIES]
        cat_urg_interaction  = [(1 if category == c else 0) * has_urgent   for c in CATEGORIES]

        return np.array([
            compound, scores['pos'], scores['neg'], scores['neu'],   # 4
            is_positive, is_negative, is_neutral,                    # 3
            has_urgent, urgent_count,                                # 2
            text_length, exclamation, question, neg_int_count,       # 4
            *cat_vec,              # 5
            *cat_neg_interaction,  # 5
            *cat_urg_interaction,  # 5  → total 28
        ], dtype=np.float32)

    def _rule_label(self, text, category):
        """Enhanced rule-based label used for synthetic data generation."""
        sia = self._get_sia()
        compound = sia.polarity_scores(str(text))['compound']
        text_lower = str(text).lower()
        has_urgent = any(kw in text_lower for kw in URGENT_KEYWORDS)

        if has_urgent:
            return "High"

        threshold = CATEGORY_BOOST_RULES.get(category)
        if threshold is not None and compound <= threshold:
            return "High"

        if compound < -0.3:
            return "High"
        elif compound < 0.1:
            return "Medium"
        else:
            return "Low"

    # ── Synthetic training data ────────────────────────────────────────────────

    def _generate_training_data(self):
        """
        Returns (texts, labels, categories) for the synthetic training corpus.
        Labels are derived from the enhanced rule-based logic so the RF learns
        to generalise beyond hard-coded thresholds.
        """
        templates = {
            "Account Access": {
                "High": [
                    "I've been locked out of my account for 3 days and nobody is helping!",
                    "Someone hacked my account and I can't log in anymore!",
                    "My account was compromised! I need urgent help immediately!",
                    "I cannot access my account at all. This is a security emergency.",
                    "Unauthorized login detected on my account. I'm very worried.",
                    "My account has been breached! Please escalate NOW!",
                    "Authentication is completely broken. I have lost access to everything.",
                    "Password reset keeps failing and I am desperate for help.",
                    "Hacker took over my account. This is an emergency!",
                    "Cannot log in after suspicious activity. My account may be stolen.",
                    "Two-factor auth is broken and someone unauthorized is inside my account.",
                    "I am locked out and my data is at risk. Please help immediately.",
                ],
                "Medium": [
                    "My password reset email isn't arriving. Can you look into it?",
                    "I'm having trouble logging in. Can you help me out?",
                    "The two-factor authentication is not working properly.",
                    "I forgot my password and need to reset it.",
                    "My login is not working after the recent update.",
                    "Having issues with authentication on my mobile device.",
                    "Cannot sign in to the app today for some reason.",
                    "My credentials don't seem to be recognized by the system.",
                    "Account access issue. Can you check what happened?",
                ],
                "Low": [
                    "Just wanted to check how to update my email address on the account.",
                    "How do I change my profile settings and preferences?",
                    "I'd like to update my account information at some point.",
                    "Quick question about my account customization options.",
                    "Is there a way to add a secondary email to my account?",
                    "Love the platform! How do I access account settings?",
                ]
            },
            "Billing Issue": {
                "High": [
                    "I was charged twice this month and nobody is responding! This is fraud!",
                    "Unauthorized charge appeared on my card. I'm calling the police!",
                    "You've stolen money from me! I want a lawyer involved right now!",
                    "Fraudulent charge on my account. I need this resolved ASAP!",
                    "I was billed $500 without my authorization. This is completely illegal!",
                    "Chargeback has been filed. This billing practice is totally unacceptable!",
                    "I've been overcharged and I am absolutely furious. This is theft!",
                    "Unauthorized subscription charge appeared on my credit card statement.",
                    "I want to cancel immediately and get a full refund right now. I am furious!",
                    "Wrong amount charged to my account. This is terrible and fraudulent!",
                    "Double-charged for a service I cancelled months ago. Lawsuit incoming!",
                    "My card was charged without consent. I am reporting this to the bank.",
                ],
                "Medium": [
                    "I noticed a discrepancy in my recent invoice amount.",
                    "I have a question about the charges on my last billing statement.",
                    "Why was I charged differently this month compared to last month?",
                    "Could you clarify the various fees listed on my latest invoice?",
                    "I'm a bit confused about the pricing breakdown for my plan.",
                    "The bill amount doesn't match the amount I expected to pay.",
                    "Having trouble understanding my subscription costs this cycle.",
                    "Can I get an itemized receipt for my most recent payment?",
                    "My billing cycle seems off. Can someone look into it?",
                ],
                "Low": [
                    "Just checking if my payment went through successfully this month.",
                    "Can I get a copy of my invoice for tax filing purposes?",
                    "When is my next scheduled billing date?",
                    "I'd like to know more about my current subscription plan details.",
                    "Just wondering about what payment methods are accepted.",
                    "Happy customer here! When does my invoice renew?",
                ]
            },
            "Technical Issue": {
                "High": [
                    "The app keeps crashing and I have completely lost all my data! This is an emergency!",
                    "The system is completely down. Our production environment is totally broken!",
                    "Critical bug is causing data corruption. We need an urgent fix right now!",
                    "Application crash is wiping user data. This is an absolute disaster!",
                    "Complete system failure on our end. We cannot function at all today!",
                    "The software is broken and I have an important deadline tomorrow morning!",
                    "Data sync failure causing total loss of a week of work. Urgent fix needed!",
                    "Platform has been down for 6 hours straight. All business operations stopped!",
                    "Security vulnerability discovered. System has been breached. Fix this immediately!",
                    "App is completely not working at all. Totally broken. Please help ASAP!",
                    "Our database crashed and we cannot recover data. Emergency situation!",
                    "Nothing loads, everything errors out. Business is at a complete standstill!",
                ],
                "Medium": [
                    "The application has been running slower than usual lately.",
                    "There seems to be a bug in the reporting feature I use often.",
                    "My data isn't syncing properly between my phone and desktop.",
                    "Getting occasional error messages while using the application.",
                    "The latest update broke one of the features I rely on frequently.",
                    "Performance has degraded quite a bit since the recent update.",
                    "Some features are not loading properly for me today.",
                    "Having intermittent connection issues with the platform this week.",
                    "The export functionality seems to be broken sometimes.",
                ],
                "Low": [
                    "Could you improve the dark mode feature? It would be nice.",
                    "Minor UI bug: the tooltip overlaps slightly with the button.",
                    "The loading animation could look a bit smoother.",
                    "Would love a keyboard shortcut for this common feature.",
                    "Just noticed a small typo in the help documentation section.",
                    "The export feature could run a tiny bit faster.",
                    "Great app overall! Just one small visual glitch to report.",
                ]
            },
            "Refund Request": {
                "High": [
                    "I demand a full refund immediately or I am taking this to court!",
                    "I have been waiting weeks for my refund. This is outright fraud!",
                    "I want my money back NOW. This treatment is completely unacceptable!",
                    "My refund was denied unfairly. I'm contacting my lawyer today!",
                    "I was charged for a cancelled subscription! Issue a refund immediately!",
                    "Your product is broken and I demand a full refund ASAP!",
                    "I was seriously misled into purchasing this. I demand my money returned!",
                    "Refund pending for 2 months now. This is absolute theft and I'm furious!",
                    "I cancelled within the trial period and still got charged. Refund NOW!",
                ],
                "Medium": [
                    "I'd like to formally request a refund for my recent purchase please.",
                    "I cancelled my subscription but haven't received a refund yet.",
                    "Can you process a refund for my unused subscription this month?",
                    "The product didn't meet my expectations at all. Requesting refund.",
                    "How long does the refund processing usually take on your end?",
                    "I returned the item last week but haven't seen a credit appear yet.",
                    "Still waiting on my refund from 2 weeks ago. Any update?",
                ],
                "Low": [
                    "Just checking on the status of my refund request from last week.",
                    "Would like to return an item that's within your return policy period.",
                    "Is it possible to get store credit instead of a cash refund?",
                    "Just following up on a previous refund request I submitted.",
                    "Happy to wait, just wanted to confirm the refund was received.",
                ]
            },
            "General Inquiry": {
                "High": [
                    "I've been a loyal customer for 10 years and this is the WORST service I've ever experienced!",
                    "This is completely unacceptable! I want to speak to a senior manager right NOW!",
                    "I am absolutely furious. Your service has completely and utterly failed me today!",
                    "This is a critical emergency. I need an immediate escalated response right now!",
                    "Urgent: My entire team of 50 people cannot work because of your platform failure!",
                    "You have lost a customer forever. This incompetence is unbelievable and infuriating!",
                ],
                "Medium": [
                    "I have some questions about your various service features and offerings.",
                    "What are your support hours and typical response times for tickets?",
                    "Can you explain the differences between your various subscription plans?",
                    "I'd like to know more about upgrading my current account to a higher tier.",
                    "Having a bit of trouble finding relevant information on your website.",
                    "What third-party integrations do you support in your platform?",
                    "How do I export all my data from your platform if I need to?",
                    "I'm having a mediocre experience and would like some help improving it.",
                ],
                "Low": [
                    "Just wanted to say your service has been absolutely great lately!",
                    "Happy customer here checking in with a minor feature question.",
                    "Love the product! Just a quick question about some customization.",
                    "Everything is working perfectly! Just one small curious question.",
                    "Thanks for the great support last time. Have one more small question.",
                    "Your team has been so wonderful and helpful. Just a small inquiry.",
                    "I'm very happy with the overall service. Just a tiny question here.",
                    "Great platform! Would love to know about any upcoming new features.",
                    "Absolutely satisfied with everything. Just checking one small detail.",
                ]
            }
        }

        # Additional cross-category edge cases
        extra = [
            ("I was hacked AND charged fraudulently. This is a disaster!", "Account Access", "High"),
            ("Unauthorized charges AND cannot log into my account!", "Billing Issue", "High"),
            ("App crashed and deleted all my billing history!", "Technical Issue", "High"),
            ("I'm taking legal action if my refund isn't processed today!", "Refund Request", "High"),
            ("System is down and no one is responding to urgent support tickets!", "General Inquiry", "High"),
            ("This is an emergency - I've been locked out and charged incorrectly!", "Account Access", "High"),
            ("Someone compromised my account and made unauthorized purchases!", "Billing Issue", "High"),
            ("I'm not entirely happy with the response time I've been getting.", "General Inquiry", "Medium"),
            ("The product is okay for now but could definitely be better.", "Technical Issue", "Medium"),
            ("Still waiting for my refund. It's been about a week so far.", "Refund Request", "Medium"),
            ("My account isn't syncing correctly. Could use some help soon.", "Account Access", "Medium"),
            ("Invoice appears to be slightly off. Please verify and correct.", "Billing Issue", "Medium"),
            ("Just curious about any exciting new features coming up soon!", "General Inquiry", "Low"),
            ("Love your product! How do I generate and export detailed reports?", "Technical Issue", "Low"),
            ("Happy long-time user here. Just checking my next billing date.", "Billing Issue", "Low"),
            ("Everything is perfect! Just asking how to share my account with a colleague.", "Account Access", "Low"),
            ("Really pleased with service. When does my refund eligibility expire?", "Refund Request", "Low"),
            ("Thanks for all the help! Just one more tiny question about features.", "General Inquiry", "Low"),
        ]

        texts, labels, categories = [], [], []
        for cat, pdict in templates.items():
            for priority, txts in pdict.items():
                for txt in txts:
                    texts.append(txt)
                    labels.append(priority)
                    categories.append(cat)

        for txt, cat, priority in extra:
            texts.append(txt)
            labels.append(priority)
            categories.append(cat)

        return texts, labels, categories

    # ── Train / Save / Load ───────────────────────────────────────────────────

    def train(self, extra_texts=None, extra_labels=None, extra_categories=None):
        """
        Train the RandomForest priority classifier.
        Base training data is synthetic; caller may supply additional labelled samples.
        """
        texts, labels, categories = self._generate_training_data()

        if extra_texts:
            texts      += list(extra_texts)
            labels     += list(extra_labels)
            categories += list(extra_categories)

        print(f"[PriorityClassifier] Training on {len(texts)} samples...")
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  {u}: {c} samples")

        X = np.array([self.extract_features(t, c) for t, c in zip(texts, categories)])
        y_enc = self.label_encoder.transform(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        self.train_metrics = {
            'accuracy': float(acc),
            'n_samples': len(texts),
            'classes': self.label_encoder.classes_.tolist()
        }
        print(f"[PriorityClassifier] Accuracy: {acc:.4f}")
        print(classification_report(
            y_test, preds, target_names=self.label_encoder.classes_
        ))
        return self.train_metrics

    def save(self, base_dir='.'):
        path = os.path.join(base_dir, 'priority_classifier.pkl')
        with open(path, 'wb') as f:
            pickle.dump({
                'model':         self.model,
                'label_encoder': self.label_encoder,
                'is_trained':    self.is_trained,
                'train_metrics': self.train_metrics,
            }, f)
        print(f"[PriorityClassifier] Saved → {path}")

    def load(self, base_dir='.'):
        path = os.path.join(base_dir, 'priority_classifier.pkl')
        if not os.path.exists(path):
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model         = data['model']
        self.label_encoder = data['label_encoder']
        self.is_trained    = data.get('is_trained', True)
        self.train_metrics = data.get('train_metrics', {})
        print(f"[PriorityClassifier] Loaded from {path}")
        return True

    # ── Category boost (post-model override) ──────────────────────────────────

    def _apply_category_boost(self, ml_priority, category, compound, has_urgent):
        """
        Post-prediction override layer.
        Can upgrade the ML prediction to High, but NEVER downgrades.
        """
        if ml_priority == "High":
            return "High", None   # already highest, no override needed

        if has_urgent:
            return "High", "urgent_keyword"

        threshold = CATEGORY_BOOST_RULES.get(category)
        if threshold is not None and compound <= threshold:
            return "High", f"category_boost({category})"

        return ml_priority, None

    # ── Public prediction API ─────────────────────────────────────────────────

    def predict(self, text, category):
        """Simple priority prediction returning just the label string."""
        if not self.is_trained:
            return self._rule_label(text, category)

        feats = self.extract_features(text, category)
        enc   = self.model.predict([feats])[0]
        ml_p  = self.label_encoder.inverse_transform([enc])[0]

        sia      = self._get_sia()
        compound = sia.polarity_scores(str(text))['compound']
        has_urg  = any(kw in str(text).lower() for kw in URGENT_KEYWORDS)

        final_p, _ = self._apply_category_boost(ml_p, category, compound, has_urg)
        return final_p

    def predict_with_explanation(self, text, category):
        """
        Returns (priority, explanation_str, confidence_dict).

        confidence_dict keys: 'High', 'Medium', 'Low'  → float probabilities
        explanation_str: human-readable string describing the decision driver
        """
        if not self.is_trained:
            priority = self._rule_label(text, category)
            return priority, "Rule-based fallback (model not yet trained)", \
                   {"High": 0.0, "Medium": 0.0, "Low": 0.0}

        sia = self._get_sia()
        vader_scores = sia.polarity_scores(str(text))
        compound     = vader_scores['compound']
        text_lower   = str(text).lower()
        urgent_hits  = [kw for kw in URGENT_KEYWORDS if kw in text_lower]
        has_urgent   = bool(urgent_hits)

        feats      = self.extract_features(text, category)
        enc        = self.model.predict([feats])[0]
        proba      = self.model.predict_proba([feats])[0]
        ml_priority = self.label_encoder.inverse_transform([enc])[0]

        # Map probabilities to class names
        confidence = {
            cls: float(p)
            for cls, p in zip(self.label_encoder.classes_, proba)
        }

        final_priority, override_reason = self._apply_category_boost(
            ml_priority, category, compound, has_urgent
        )

        # ── Build explanation string ──────────────────────────────────────────
        if override_reason == "urgent_keyword":
            explanation = (
                f"⚡ Escalated to High — urgent keyword(s) detected: "
                f"**{', '.join(urgent_hits[:3])}**"
            )
        elif override_reason and override_reason.startswith("category_boost"):
            threshold = CATEGORY_BOOST_RULES.get(category, "N/A")
            explanation = (
                f"⚡ Escalated to High — category boost: "
                f"**{category}** + negative sentiment "
                f"(score {compound:.3f} ≤ threshold {threshold})"
            )
        else:
            top_class  = max(confidence, key=confidence.get)
            top_conf   = confidence[top_class]
            explanation = (
                f"🤖 ML decision — {top_class} at {top_conf:.1%} confidence. "
                f"VADER compound: {compound:+.3f}"
            )
            if has_urgent:
                explanation += f" | ⚡ Keyword boost applied: {', '.join(urgent_hits[:2])}"

        return final_priority, explanation, confidence
