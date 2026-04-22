# Configurazione delle regole di scoring per bollette elettriche

# Campi da usare per identificare righe analoghe tra diversi risultati
# Questi campi vengono usati per capire se due periodi si riferiscono allo stesso consumo
UNIQUE_IDENTIFIERS = ['codice', 'giorno_inizio', 'giorno_fine']

# Campi da ignorare nel conteggio dei punten tijdens het vergelijken
# Deze velden dragen niet bij aan de eindscore
IGNORED_FIELDS = {'timestamp', 'indirizzo', 'consumo_annuale'}
