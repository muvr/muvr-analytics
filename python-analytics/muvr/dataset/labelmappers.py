"""A collection of default label mappers"""

def inverse_mapping(mapping):
    return dict((label, ex) for (ex, labels) in mapping.iteritems() for label in labels)    


def generate_activity_labelmapper():
    exercise_mapping = {
        "E": ["biceps-curl", "lateral-raise", "lateral-raise", "triceps-extension"],
        "-": ["walking", ""]
    }

    inv_exercise_map = inverse_mapping(exercise_mapping)
    
    return lambda label: inv_exercise_map.get(label)


def generate_exercise_labelmapper():
    exercise_mapping = {
        "arms/biceps-curl": [
            "biceps curls (left)", 
            "biceps-curl", "bice", 
            "arms/biceps-curl", 
            "BC ", "bicep curls", 
            "bicep"],
        "shoulders/lateral-raise": [
            "arms/lateral-raise", 
            "lateral raises", 
            "lateral", 
            "lateral-raise", 
            "LR", 
            "LR "],
        "arms/triceps-extension": [
            "triceps-extension	", 
            "tc ", 
            "TE", 
            "tc", 
            "triceps-extension"],
        "arms/triceps-dips": [
            "triceps-dips"
        ]
        
    }

    inv_exercise_map = inverse_mapping(exercise_mapping)

    return lambda label: inv_exercise_map.get(label)
