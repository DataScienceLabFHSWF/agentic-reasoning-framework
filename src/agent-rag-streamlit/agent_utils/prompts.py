"""
prompts.py
Promptvorlagen für den RAG-Chat-Workflow
"""

from langchain_core.prompts import ChatPromptTemplate


ROUTER_PROMPT = ChatPromptTemplate.from_template("""
Du analysierst, warum die abgerufenen Dokumente den Relevanzschwellenwert für eine RAG-Anfrage nicht erreicht haben.

Anfrage: {query}
Maximaler Relevanzwert: {max_score}
Schwellenwert: {threshold}

Die Dokumente wurden als nicht relevant genug eingestuft. Erstelle eine kurze Analyse, warum die abgerufenen Dokumente
möglicherweise nicht geeignet sind, um diese Anfrage zu beantworten. Dies dient dazu, die allgemeine Antwort zu verbessern.

Halte deine Analyse prägnant und sachlich.
""")


INTENT_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_template("""
Du bist ein Assistent zur Intentionsklassifikation für ein spezialisiertes RAG-System über deutsche Kerntechnik und Nuklearsicherheit.

DOKUMENTENÜBERSICHT:
Unsere Wissensdatenbank enthält hauptsächlich:
- Sicherheitsberichte kerntechnischer Anlagen (z.B. KRB II - Kernkraftwerk Gundremmingen)
- Atomrechtliche Bestimmungen und Genehmigungsverfahren
- Technische Spezifikationen von Kernkraftwerken (Reaktortypen, Leistung, Blöcke)
- Stilllegungsverfahren und Rückbau kerntechnischer Anlagen
- Strahlenschutz und Überwachungsmaßnahmen
- Zwischen- und Endlagerung radioaktiver Abfälle
- Genehmigungsinhaber und Betreiber deutscher Kernkraftwerke
- Technische Sicherheitssysteme und Notfallmaßnahmen

CHAT-VERLAUF (falls vorhanden):
{chat_history}

AUFGABE:
Analysiere die folgende Benutzeranfrage und klassifiziere sie:

Benutzeranfrage: {query}

KLASSIFIZIERUNGSOPTIONEN:
1. FOLLOW_UP: Wenn die Anfrage eine Nachfrage zu einem vorherigen Thema ist (z.B. "Was bedeutet das?", "Erzähl mir mehr darüber", "Wie funktioniert das?", "Welche Details gibt es dazu?")
2. RELEVANT: Neue Fragen zu deutschen Kernkraftwerken, Atomrecht, Nuklearsicherheit, Genehmigungen, Stilllegung, Atommüll
3. NOT RELEVANT: Allgemeine Gespräche, andere Energieformen, ausländische Nuklearanlagen, grundlegende Physik ohne Nuklearbezug

Antworte mit "FOLLOW_UP", "RELEVANT" oder "NOT RELEVANT" und gib eine kurze Begründung.

Antwort:
""")

FOLLOW_UP_PROMPT = ChatPromptTemplate.from_template("""
Du beantwortest eine Nachfrage basierend auf dem vorherigen Gesprächskontext über deutsche Kernkraftwerke und Nuklearsicherheit.

VORHERIGER KONTEXT:
{previous_context}

AKTUELLE NACHFRAGE:
{query}

ANWEISUNGEN:
1. Verwende den vorherigen Kontext, um die Nachfrage zu beantworten
2. Falls der Kontext nicht ausreicht, gib das ehrlich zu und schlage vor, eine spezifischere Frage zu stellen
3. Antworte ausschließlich auf Deutsch
4. Beziehe dich direkt auf die vorherigen Informationen
5. Sei hilfreich und ermutige weitere Fragen

Antwort:
""")

HUMAN_INTERVENTION_PROMPT = ChatPromptTemplate.from_template("""
Der Benutzer möchte den Workflow manuell steuern.

Benutzerbefehl: {command}
Aktuelle Anfrage: {query}

Verfügbare Optionen:
- "force_rag": Erzwinge RAG-Suche auch bei niedriger Relevanz
- "use_context": Verwende nur den Chat-Verlauf ohne neue Suche
- "general": Behandle als allgemeine Frage

Erkläre dem Benutzer, welche Option gewählt wurde und warum.

Antwort:
""")

GENERAL_PROMPT = ChatPromptTemplate.from_template("""
Der Nutzer hat eine Frage gestellt, aber die Anfrage ist nicht relevant für unsere spezialisierte Nuklear-Wissensdatenbank, oder die abgerufenen Dokumente waren nicht relevant genug.

Benutzeranfrage: {query}
Kontext: {context}

ANWEISUNGEN:
1. Falls die Anfrage nicht nuklearspezifisch ist: Erkläre höflich, dass du auf deutsche Kernkraftwerke und Nuklearsicherheit spezialisiert bist
2. Falls die Dokumente nicht relevant genug waren: Schlage vor, die Frage spezifischer zu formulieren mit Begriffen wie:
   - Spezifische Anlagennamen (z.B. "KRB II", "Gundremmingen")
   - Atomrechtliche Begriffe (z.B. "Genehmigung", "Stilllegung", "Rückbau")
   - Technische Aspekte (z.B. "Reaktortyp", "Leistung", "Sicherheitssysteme")
3. Biete Beispiele für relevante Fragen an

Sei höflich, hilfreich und ermutige den Nutzer, spezifischere nuklearbezogene Fragen zu stellen.

Antwort:
""")

SUMMARIZER_PROMPT = ChatPromptTemplate.from_template("""
Basierend auf den folgenden, als hoch relevant eingestuften Dokumenten (Relevanzwert über dem Schwellenwert) 
erstelle bitte eine prägnante und genaue Antwort auf die Frage des Nutzers auf Deutsch.

Frage des Nutzers: {query}

Relevante Dokumente:
{context}

Anweisungen:
- Antworte ausschließlich auf Deutsch
- Die Antwort basiert auf den Informationen in den Dokumenten
- Die Dokumente wurden als relevant für die Anfrage bestätigt
- Sei sachlich und zitiere spezifische Informationen aus den Dokumenten, wenn möglich
- Halte die Antwort dennoch gesprächig und hilfreich
- Verwende deutsche Fachbegriffe für nukleartechnische Konzepte

Antwort:
""")