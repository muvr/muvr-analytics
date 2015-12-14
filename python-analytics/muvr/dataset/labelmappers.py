"""A collection of default label mappers"""

exercise_mapping = {
    "biceps-curl": [
        "biceps curls (left)",
        "biceps-curl",
        "bice",
        "arms/biceps-curl",
        "bc",
        "bicep curls",
        "bicep"],
    "lateral-raise": [
        "arms/lateral-raise",
        "lateral raises",
        "lateral",
        "lateral-raise",
        "lr"],
    "triceps-extension": [
        "te",
        "tc",
        "triceps-extension"],
    "barbell-biceps-curl": [
        "barbell-curl",
        "barbell-biceps-curl"],
    "angle-chest-press": [
        "angle-chest-press"],
    "dumbbell-bench-press": [
        "dumbbell-chest-press",
        "dumbbell-bench-press"],
    "dumbbell-shoulder-press": [
        "dumbbell shoulder press",
        "dumbbell-shoulder-press"],
    # "vertical-swing": [
    #     "vertical swing",
    #     "vertical-swing"],
    "triceps-pushdown": [
        "triceps-pushdown",
        "rope-tricep-pushdown"],
    # "barbell-squat": [
    #     "barbell-squat"],
    "lateral-pulldown-straight": [
        "lateral-pulldown-straight",
        "lat-pulldown-straight"],
    "triceps-dips": [
        "triceps-dips"],
    # "bent-arm-barbell-pullover": [
    #     "bent-arm-barbell-pullover"],
    "bench-press": [
        "bench-press"
    ]
    # "running-machine-hit": [
    #     "running-machine-hit",
    #     "running-machine-hiit",
    #     "hiit",
    #     "hiit running machine"]
}

def inverse_mapping(mapping):
    return dict((label, ex) for (ex, labels) in mapping.iteritems() for label in labels)    


def preprocess_label(label):
    return label.lower().strip()


def generate_activity_labelmapper():
    activity_mapping = {
        "E": [label for labels in exercise_mapping.values() for label in labels],
        "-": ["walking", ""]
    }

    inv_exercise_map = inverse_mapping(activity_mapping)
    
    return lambda label: inv_exercise_map.get(preprocess_label(label))


def generate_exercise_labelmapper():
    inv_exercise_map = inverse_mapping(exercise_mapping)

    return lambda label: inv_exercise_map.get(preprocess_label(label))
