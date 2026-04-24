# Configurazione delle regole di scoring per registri rifiuti

# Campi che identificano univocamente un movimento (CER + Quantità)
UNIQUE_IDENTIFIERS = ['codice_cer', 'quantita']

# Campi da ignorare nel conteggio dei punti
IGNORED_FIELDS = {'timestamp', 'anno'}
