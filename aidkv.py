import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# 1. Inserisci qui la tua API Key di Google AI Studio
os.environ["GOOGLE_API_KEY"] = "AIzaSyAvN6Ra9EcgNATiuA3p78VYM6NmqF-dHvo"

# 2. Carica il dataset DKV che la tua mappa ha generato
file_csv = "dataset_machine_learning_dkv.csv"

try:
    df = pd.read_csv(file_csv, sep=';')
    print(f"✅ Dati caricati con successo: {len(df)} righe pronte per l'analisi.")
except FileNotFoundError:
    print(f"❌ Errore: File {file_csv} non trovato. Fai prima girare lo script della mappa!")
    exit()

# 3. Inizializza il "cervello" usando Google Gemini 1.5 Flash (Veloce e Gratuito)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# 4. Crea l'Agente
# allow_dangerous_code=True serve per far eseguire il codice Python in locale a LangChain
agente = create_pandas_dataframe_agent(
    llm, 
    df, 
    verbose=True, # Mostra i ragionamenti nel terminale
    allow_dangerous_code=True 
)

# 5. Avvia la chat
print("\n🤖 WEX Copilot Logistico Avviato (Powered by Gemini)!")
print("Scrivi 'esci' per terminare.\n")

while True:
    domanda = input("Tu (Logistica): ")
    if domanda.lower() in ['esci', 'quit', 'exit', 'basta']:
        print("🤖 Chiusura sistema...")
        break
        
    if not domanda.strip():
        continue
        
    try:
        risposta = agente.invoke(domanda)
        print(f"\n🤖 Copilot: {risposta['output']}\n")
    except Exception as e:
        print(f"\n⚠️ Si è verificato un errore: {e}\n")