# =====================================================
# 💡 App Gradio basada en tu notebook original
# =====================================================
# Ejecuta cada celda en orden dentro de Google Colab
# Al final obtendrás un enlace público (.gradio.live)
# =====================================================

# 1️⃣ Instalar dependencias
!pip install gradio sentence-transformers pandas openpyxl transformers mlxtend

# 2️⃣ Importar librerías y cargar datos / modelo
import gradio as gr
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline # Import pipeline for LLM
import numpy as np # Import numpy
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# Cargar el inventario desde tu Google Drive
df_inv = pd.read_excel("/content/drive/MyDrive/proyecto ia/inventario aku para IA.xlsx", skiprows=1, header=0) # Added skiprows and header
df_inv.columns = ["CODIGO", "PRODUCTO", "CATEGORIA", "PRECIO", "STOCK"] # Added column renaming
df_inv['__PRODUCTO__'] = df_inv['PRODUCTO'].astype(str) # Added __PRODUCTO__ creation

# Load ventas data for recommendations
df_ven = pd.read_excel("/content/drive/MyDrive/proyecto ia/ventas aku IA.xlsx", skiprows=1, header=0)
df_ven.columns = ["CODIGO","PRODUCTO", "FECHA", "CANTIDAD", "TOTAL"]
df_ven['__PRODUCTO__'] = df_ven['PRODUCTO'].astype(str)


# Prepare text for classification (create text column) - Redefine this here for the Gradio app
def detect_product_col(df):
    candidates = ["Producto","producto","Nombre","NAME","Item","item","nombre_producto","Producto Nombre"]
    for c in df.columns:
        if c in candidates:
            return c
    return df.columns[0]

prod_col_inv = detect_product_col(df_inv)

# Create __TEXT__ in inventory (base for the classifier)
# Heuristically choose description columns
desc_cols = [c for c in df_inv.columns if isinstance(c, str) and ('desc' in c.lower() or 'descripcion' in str(c).lower() or 'detalle' in str(c).lower())]

df_inv['__TEXT__'] = df_inv['__TEXT__'] = df_inv.apply(lambda row: ' '.join([str(row[c]) for c in desc_cols if c in df_inv.columns and pd.notnull(row[c])]), axis=1).str.strip()
df_inv['__TEXT__'] = df_inv['__PRODUCTO__'].astype(str) + ". " + df_inv['__TEXT__']


# Attempt to load the trained model
clf = mlb = embedder = None # Initialize to None
try:
    with open("artifact_clf.pkl", "rb") as f:
        art = pickle.load(f)
    clf = art["clf"]
    mlb = art["mlb"]
    embedder = SentenceTransformer(art.get("embedder_name", "all-MiniLM-L6-v2"))
    print("✅ Modelo de clasificación cargado correctamente.")
except Exception as e:
    print("⚠️ No se encontró el modelo entrenado para clasificación:", e)


# Attempt to load the LLM
gen = None # Initialize LLM pipeline to None
use_llm = False
try:
    # Check if a pipeline is already loaded to avoid reloading
    # This check might be tricky in Gradio's stateless nature. Re-loading might be necessary.
    # Let's try loading it directly here.
    gen = pipeline("text-generation", model="sshleifer/tiny-gpt2")
    use_llm = True
    print("✅ LLM (tiny-gpt2) cargado correctamente.")
except Exception as e:
    print("⚠️ No se cargó LLM small; se usará plantilla. Detalle:", e)
    use_llm = False # Ensure use_llm is False if loading fails


# Helper function to predict labels
def predecir_etiquetas(texts):
    # texts: list of strings
    if clf is None or embedder is None or mlb is None:
        return ["Modelo no cargado"] * len(texts)
    try:
        embs = embedder.encode(texts, show_progress_bar=False) # Disable progress bar for Gradio
        preds = clf.predict(embs)
        # convert to readable labels
        labels = []
        for row in preds:
            labels.append([mlb.classes_[i] for i,v in enumerate(row) if v==1])
        return labels
    except Exception as e:
        print(f"Error predicting labels: {e}")
        return ["Error en predicción"] * len(texts)


# Helper function to generate post (Always use template due to tiny-gpt2 limitations)
def generar_post_template(name, description="", age=None, tone="amigable"):
    post = f"{name} — {description}"
    if age:
        post += f" Ideal para {age}."
    post += " Encuéntralo en Aku. 💛🧸"
    return post + "\n\n(Nota: Se usó una plantilla ya que el modelo de lenguaje pequeño no generó texto coherente.)"


# Helper function to generate hashtags
def generar_hashtags(categories, age_range=None, top_n=6):
    if isinstance(categories, str):
        categories = [categories]
    tags = []
    for c in categories:
        clean = str(c).strip().replace(" ","")
        if clean:
            tags.append(f"#{clean}")
    if age_range:
        tags.append(f"#Edad{str(age_range).replace(' ','')}")
    tags += ["#JuguetesEducativos","#ProteccionBebe","#AkuTienda"]
    uniq = []
    for t in tags:
        if t not in uniq:
            uniq.append(t)
    return uniq[:top_n]


# 3️⃣ Función principal (lógica de generación de post)
def generar_contenido_producto(producto):
    row = df_inv[df_inv["__PRODUCTO__"] == producto].iloc[0]

    # Get product text for classification
    product_text = row.get('__TEXT__', str(row.get('PRODUCTO', '')))

    # Generate post
    post = generar_post_template(row.get('__PRODUCTO__',''), row.get(desc_cols[0],'') if desc_cols else '', None)


    # Generate labels (hashtags)
    etiquetas = []
    if clf and embedder and mlb:
         predicted_labels = predecir_etiquetas([product_text])[0]
         etiquetas = generar_hashtags(predicted_labels)
    else:
        # Fallback hashtags if classification model is not loaded
        etiquetas = generar_hashtags([row.get('CATEGORIA', '')]) if row.get('CATEGORIA', '') else generar_hashtags([])


    # Generate combo recommendations
    recomendaciones = []
    rules_sorted_local = pd.DataFrame() # Local variable for rules

    try:
        # Generate association rules from sales data within this function
        group_col = None
        for c in df_ven.columns:
            name = str(c).lower()
            if 'fact' in name or 'pedido' in name or 'orden' in name or 'id' in name:
                group_col = c
                break

        if group_col:
            transactions = df_ven.groupby(group_col)['__PRODUCTO__'].apply(lambda s: list(s.dropna().astype(str))).tolist()
        else:
            transactions = df_ven.apply(lambda r: [str(r.get('__PRODUCTO__'))] if pd.notnull(r.get('__PRODUCTO__')) else [], axis=1).tolist()

        transactions = [t for t in transactions if len(t)>0]

        if len(transactions) > 1:
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            trans_df = pd.DataFrame(te_ary, columns=te.columns_)
            freq_items = apriori(trans_df, min_support=0.02, use_colnames=True)
            rules = association_rules(freq_items, metric="confidence", min_threshold=0.2)
            rules_sorted_local = rules.sort_values(by='confidence', ascending=False)
            print("✅ Reglas de asociación generadas localmente.")
        else:
            print("⚠️ Pocas transacciones para generar reglas de asociación localmente.")

    except Exception as e:
        print(f"⚠️ Error al generar reglas de asociación: {e}. Las recomendaciones de combos no estarán disponibles.")
        rules_sorted_local = pd.DataFrame() # Ensure it's an empty DataFrame if generation fails


    try:
        if not rules_sorted_local.empty:
             for _, r in rules_sorted_local.iterrows():
                # Ensure 'antecedents' are processed as sets of strings for accurate matching
                try:
                    # Use set(r['antecedents']) directly if they are already frozensets
                    ants = set(r['antecedents']) if isinstance(r['antecedents'], frozenset) else set(map(str, eval(str(r['antecedents']))))
                    cons = list(r['consequents']) if isinstance(r['consequents'], frozenset) else list(map(str, eval(str(r['consequents'])))) # Corrected: added closing parenthesis

                except Exception as e:
                    print(f"Error parsing rule antecedents/consequents: {e}")
                    continue # Skip this rule if parsing fails

                if str(producto) in ants:
                    recomendaciones += cons
    except Exception as e:
        print(f"Error processing recommendations in Gradio: {e}")
        pass # Continue if there's an error

    combos = ", ".join(list(dict.fromkeys(recomendaciones))) if recomendaciones else "No hay recomendaciones de combos."


    return (
        f"🧾 **Post sugerido:**\n\n{post}",
        f"🏷️ **Etiquetas recomendadas:**\n{' '.join(etiquetas)}",
        f"🛍️ **Combos sugeridos:**\n{combos}",
    )

# 4️⃣ Crear interfaz con Gradio
def interfaz_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("## 🤖 Aku IA — Generador de Posts para Redes Sociales")
        gr.Markdown(
            "Selecciona un producto de tu inventario y obtén un post optimizado para redes sociales, "
            "junto con etiquetas y recomendaciones de combos."
        )

        producto = gr.Dropdown(
            choices=df_inv["__PRODUCTO__"].tolist(),
            label="Selecciona un producto",
        )

        boton = gr.Button("✨ Generar post y recomendaciones")
        salida_post = gr.Markdown(label="Post Sugerido")
        salida_tags = gr.Markdown(label="Etiquetas Recomendadas")
        salida_combos = gr.Markdown(label="Combos Sugeridos")

        boton.click(generar_contenido_producto, inputs=producto, outputs=[salida_post, salida_tags, salida_combos])

    return demo

demo = interfaz_gradio()

# 5️⃣ Ejecutar la app con enlace público
if __name__ == "__main__":
    demo.launch(share=True)
