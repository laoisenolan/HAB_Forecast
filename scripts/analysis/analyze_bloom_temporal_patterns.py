import pandas as pd

# Load the integrated data
data = pd.read_csv('cache/conservative_pipeline/01_integrated_data.csv')
data['date'] = pd.to_datetime(data['date'])

print('=== TEMPORAL HAB DISTRIBUTION ANALYSIS ===')
print(f'Total records: {len(data)}')
print(f'Date range: {data["date"].min()} to {data["date"].max()}')
print()

# Analyze HAB events by year
print('HAB Events by Year:')
for year in range(2017, 2024):
    year_data = data[data['date'].dt.year == year]
    if len(year_data) > 0:
        hab_karenia = year_data['hab_karenia_mikimotoi'].sum()
        hab_hetero = year_data['hab_heterosigma_akashiwo'].sum()
        hab_chat = year_data['hab_chattonella_antiqua'].sum()
        hab_coch = year_data['hab_cochlodinium_polykrikoides'].sum()
        total_hab = hab_karenia + hab_hetero + hab_chat + hab_coch
        print(f'{year}: {len(year_data):4d} records, {total_hab:4d} total HAB events')
        print(f'       Karenia: {hab_karenia:3d}, Heterosigma: {hab_hetero:3d}, Chattonella: {hab_chat:3d}, Cochlodinium: {hab_coch:3d}')

print()
print('=== TEMPORAL SPLIT ANALYSIS ===')
split_date = '2021-09-17'
train_data = data[data['date'] < split_date]
test_data = data[data['date'] >= split_date]

print(f'Split date: {split_date}')
print(f'Train period: {train_data["date"].min()} to {train_data["date"].max()} ({len(train_data)} records)')
print(f'Test period: {test_data["date"].min()} to {test_data["date"].max()} ({len(test_data)} records)')

print()
print('Train set HAB events:')
for species in ['hab_karenia_mikimotoi', 'hab_heterosigma_akashiwo', 'hab_chattonella_antiqua', 'hab_cochlodinium_polykrikoides']:
    pos_count = train_data[species].sum()
    print(f'  {species}: {pos_count}')

print()
print('Test set HAB events:')
for species in ['hab_karenia_mikimotoi', 'hab_heterosigma_akashiwo', 'hab_chattonella_antiqua', 'hab_cochlodinium_polykrikoides']:
    pos_count = test_data[species].sum()
    print(f'  {species}: {pos_count}')

print()
print('=== DATA QUALITY ANALYSIS ===')
print('Missing values in core features:')
core_features = ['sst', 'chlor_a', 'ocean_salinity', 'wind_u', 'wind_v', 'turbidity_index']
for feature in core_features:
    if feature in data.columns:
        missing = data[feature].isnull().sum()
        percentage = (missing / len(data)) * 100
        print(f'  {feature}: {missing} missing ({percentage:.1f}%)')
