import pandas as pd
import numpy as np
import re
from math import radians, cos, sin, asin, sqrt

# -------------------------
# FUNCIONES UTILITARIAS
# -------------------------

def convertir_hora_a_periodo(hora):
    if 0 <= hora <= 5:
        return 'Madrugada'
    elif 6 <= hora <= 11:
        return 'Mañana'
    elif hora == 12:
        return 'Medio día'
    elif 13 <= hora <= 17:
        return 'Tarde'
    elif 18 <= hora <= 23:
        return 'Noche'
    return 'Desconocido'

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return R * (2 * asin(sqrt(a)))

def card_type(cc_num):
    cc = str(cc_num)
    if cc.startswith('4') and len(cc) in [13, 16, 19]:
        return 'Visa'
    elif cc.startswith(('5155', '2221', '2720')):
        return 'MasterCard'
    elif cc.startswith(('34', '37')):
        return 'American Express'
    elif cc.startswith(('6011', '622', '64', '65')):
        return 'Discover'
    elif cc.startswith(('300', '305', '36', '38', '39')):
        return 'Dinner Club'
    return 'Unknown'

def extraer_tipo_via(direccion):
    tipos = ['Avenue', 'Street', 'Boulevard', 'Drive', 'Road', 'Lane', 'Court', 'Place', 'Parkway', 'Circle', 'Trail', 'Way', 'Terrace', 'Loop', 'Creek', 'Brook', 'Manor', 'Ridge', 'View']
    for tipo in tipos:
        if re.search(rf'\\b{tipo}\\b', direccion, re.IGNORECASE):
            return tipo
    return 'Otro'

def agrupar_ocupacion(ocupacion):
    ocup = str(ocupacion).lower()
    if any(p in ocup for p in ['nurse', 'therapist', 'psychologist', 'doctor']):
        return "Salud"
    elif 'teacher' in ocup or 'lecturer' in ocup:
        return "Educación"
    elif 'engineer' in ocup or 'developer' in ocup:
        return "Ingeniería"
    elif 'scientist' in ocup or 'chemist' in ocup:
        return "Ciencias"
    elif 'accountant' in ocup or 'banker' in ocup or 'finance' in ocup:
        return "Finanzas"
    elif 'artist' in ocup or 'writer' in ocup or 'musician' in ocup:
        return "Artes"
    elif 'manager' in ocup or 'consultant' in ocup or 'executive' in ocup:
        return "Negocios"
    elif 'architect' in ocup or 'survey' in ocup:
        return "Construcción"
    elif 'officer' in ocup or 'administrator' in ocup:
        return "Sector público"
    return "Otros"

def renombrar_estados(df):
    us_states_es = {
        'NC': 'Carolina del Norte', 'WA': 'Washington', 'ID': 'Idaho', 'MT': 'Montana',
        'VA': 'Virginia', 'PA': 'Pensilvania', 'KS': 'Kansas', 'TN': 'Tennessee',
        'IA': 'Iowa', 'WV': 'Virginia Occidental', 'FL': 'Florida', 'CA': 'California',
        'NM': 'Nuevo México', 'NJ': 'Nueva Jersey', 'OK': 'Oklahoma', 'IN': 'Indiana',
        'MA': 'Massachusetts', 'TX': 'Texas', 'WI': 'Wisconsin', 'MI': 'Míchigan',
        'WY': 'Wyoming', 'HI': 'Hawái', 'NE': 'Nebraska', 'OR': 'Oregón',
        'LA': 'Luisiana', 'DC': 'Distrito de Columbia', 'KY': 'Kentucky',
        'NY': 'Nueva York', 'MS': 'Misisipi', 'UT': 'Utah', 'AL': 'Alabama',
        'AR': 'Arkansas', 'MD': 'Maryland', 'GA': 'Georgia', 'ME': 'Maine',
        'AZ': 'Arizona', 'MN': 'Minnesota', 'OH': 'Ohio', 'CO': 'Colorado',
        'VT': 'Vermont', 'MO': 'Misuri', 'SC': 'Carolina del Sur', 'NV': 'Nevada',
        'IL': 'Illinois', 'NH': 'Nuevo Hampshire', 'SD': 'Dakota del Sur', 'AK': 'Alaska',
        'ND': 'Dakota del Norte', 'CT': 'Connecticut', 'RI': 'Rhode Island', 'DE': 'Delaware'
    }
    df['state'] = df['state'].replace(us_states_es)
    return df

# -------------------------
# FUNCIÓN PRINCIPAL
# -------------------------

def preprocesar_dataset(ruta_csv):
    df = pd.read_csv(ruta_csv)

    df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])

    df['mes'] = df['trans_date_trans_time'].dt.month
    df['año'] = df['trans_date_trans_time'].dt.year
    df['horas'] = df['trans_date_trans_time'].dt.hour
    df['horas'] = df['horas'].apply(convertir_hora_a_periodo)
    df['dias'] = df['trans_date_trans_time'].dt.day

    df['edad'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
    df['grupo_edad'] = pd.cut(df['edad'], bins=[0, 18, 25, 35, 45, 55, 65, 100], labels=['0-18', '19-25', '26-35', '36-45', '46-55', '56-65', '66+'], right=False)

    df['distancia_cliente_negocio'] = df.apply(lambda x: haversine(x['lat'], x['long'], x['merch_lat'], x['merch_long']), axis=1)

    df['cliente'] = df['first'] + ' ' + df['last']
    df['card_type'] = df['cc_num'].astype(str).apply(card_type)

    df['tipo_via'] = df['street'].apply(extraer_tipo_via)
    df.loc[df['amt'] >= 100.0, 'street'] = 'Via_Montos_Altos'
    df.loc[df['amt'] < 100.0, 'street'] = 'Via_Montos_Bajos'

    df = renombrar_estados(df)
    df['merchant'] = df['merchant'].str[6:].astype('category')
    df['job'] = df['job'].apply(agrupar_ocupacion)

    df['category'] = df['category'].astype('category')
    df['job'] = df['job'].astype('category')
    df['gender'] = df['gender'].astype('category')
    df['horas'] = df['horas'].astype('category')
    df['card_type'] = df['card_type'].astype('category')
    df['street'] = df['street'].astype('category')

    df_final = df.loc[:, ['trans_date_trans_time', 'cliente', 'gender', 'grupo_edad', 'job', 'street', 'horas', 'card_type', 'amt', 'category', 'merchant', 'distancia_cliente_negocio', 'is_fraud']]
    df_final.columns = ['Fecha_Compra','Cliente','Genero','Grupo_Edad','Trabajo','Tipo_Via','Horario','Tipo_Tarjeta','Monto_Transaccion','Categoria_Compra','Negocio','Distancia_Cliente_Negocio','Fraude']

    return df_final

# -------------------------
# EJEMPLO DE USO
# -------------------------
if __name__ == "__main__":
    ruta = "C:/Users/leo_2/Documents/tarjetas_creditos/data/cruda/fraudTrain.csv"
    df_final = preprocesar_dataset(ruta)
    df_final.to_csv("C:/Users/leo_2/Documents/tarjetas_creditos/data/procesada/data_procesada.csv", index=False)
    print("Datos procesados guardados con éxito.")
