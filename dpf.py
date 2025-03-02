def calculate_dpf(race, age, family_members_with_diabetes):
    # Define weights for race and age
    race_weights = {
        'Caucasian': 1.0,
        'African American': 1.2,
        'Hispanic': 1.1,
        'Asian': 1.3,
        'Other': 1.0
    }
    
    age_weights = {
        '0-20': 0.5,
        '21-40': 1.0,
        '41-60': 1.5,
        '61+': 2.0
    }
    
    # Check if the person has no family members with diabetes
    if family_members_with_diabetes == 0:
        return 0.1
    
    # Calculate the DPF
    race_weight = race_weights.get(race, 1.0)
    age_weight = None
    
    if age <= 20:
        age_weight = age_weights['0-20']
    elif 21 <= age <= 40:
        age_weight = age_weights['21-40']
    elif 41 <= age <= 60:
        age_weight = age_weights['41-60']
    else:
        age_weight = age_weights['61+']
    
    dpf = race_weight * age_weight * family_members_with_diabetes
    return dpf

# Example usage
race = 'Asian'
age = 35
family_members_with_diabetes = 2

dpf = calculate_dpf(race, age, family_members_with_diabetes)
print(f'Diabetes Pedigree Function: {dpf}')