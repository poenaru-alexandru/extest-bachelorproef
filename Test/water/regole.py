# Configurazione delle regole di scoring per bollette acqua

# Campi da usare per identificare righe analoghe tra diversi risultati
UNIQUE_IDENTIFIERS = ['codice', 'giorno_inizio', 'giorno_fine']

# Campi da ignorare nel conteggio dei punti durante il confronto
IGNORED_FIELDS = {'timestamp', 'indirizzo', 'consumo_annuale'}
