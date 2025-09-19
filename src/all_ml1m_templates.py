all_tasks = {}

# =====================================================
# Task Subgroup 1 -- Rating Prediction -- (Adapted for ML1M)
# =====================================================
task_subgroup_1 = {}
template = {}

# Template 1-1 (Direct rating prediction)
template['source'] = "Which star rating will user_{} give movie_{} ? (0 being lowest and 10 being highest)"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_id'] # In P5_ML1M_Dataset, these will be mapped IDs
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "1-1"
task_subgroup_1["1-1"] = template
template = {}

# Template 1-2 (Rating prediction with movie title)
template['source'] = "How will user_{} rate the movie \"{}\"? (0 being lowest and 10 being highest)"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "1-2"
task_subgroup_1["1-2"] = template
template = {}

# Template 1-3 (Predict if a user will give a specific rating - Yes/No)
template['source'] = "Will user_{} give movie_{} a {}-star rating? (0 being lowest and 10 being highest)"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'item_id', 'star_rating_to_check'] # Example: star_rating_to_check could be the actual rating or a random one
template['target_argc'] = 1
template['target_argv'] = ['yes_no'] # Target will be 'yes' or 'no'
template['id'] = "1-3"
task_subgroup_1["1-3"] = template
template = {}

# Template 1-4 (Predict like/dislike)
template['source'] = "Does user_{} like or dislike movie_{}?"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['like_dislike'] # Target: 'like' (rating >= 4) or 'dislike'
template['id'] = "1-4"
task_subgroup_1["1-4"] = template
template = {}

# Template 1-5 (Predict preference with title in prompt)
template['source'] = "Predict user_{}'s preference for movie_{} (titled \"{}\"). Choose a rating from 0 to 10."
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'item_id', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "1-5"
task_subgroup_1["1-5"] = template
template = {}

# Template 1-6 (Using user_desc - for ML1M, user_desc can be the raw UserID string)
template['source'] = "What star rating do you think user {} will give movie_{}? (0 being lowest and 10 being highest)"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "1-6"
task_subgroup_1["1-6"] = template
template = {}

# Template 1-7 (user_desc and item_title)
template['source'] = "How will user {} rate the movie \"{}\"? (0 being lowest and 10 being highest)"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "1-7"
task_subgroup_1["1-7"] = template
template = {}

# Template 1-8 (user_desc, specific star_rating check - Yes/No)
template = {}
template['source'] = "Will user {} give the movie \"{}\" a {}-star rating? (0 being lowest and 10 being highest)"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 3
template['source_argv'] = ['user_desc', 'item_title', 'star_rating_to_check']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-8"
task_subgroup_1["1-8"] = template # Add this line
template = {}

# Template 1-9 (user_desc, like/dislike)
template['source'] = "Does user {} like or dislike the movie \"{}\"?"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['like_dislike']
template['id'] = "1-9"
task_subgroup_1["1-9"] = template # Add this line
template = {}

# Template 1-10 (user_desc, direct rating prediction with title)
template['source'] = "Predict user {}'s preference towards the movie \"{}\" (0 being lowest and 10 being highest)"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "1-10"
task_subgroup_1["1-10"] = template # Add this line
template = {} # Reset for safety if more templates are added laters

all_tasks['rating'] = task_subgroup_1


task_subgroup_6 = {}
# Template 6-1 (Diversity - 0-10)
template['source'] = "How diverse is the movie_{} for user_{}? (0 being not diverse, 10 being very diverse)"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "6-1"
task_subgroup_1["6-1"] = template
template = {}

# Template 6-2 (Diversity - Three choices)
template['source'] = "For user_{}, is movie_{} significantly diversed from the movies they usually watch? (0 being lowest and 10 being highest)"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "6-2"
task_subgroup_1["6-2"] = template
template = {}

# Template 6-3 (Novelty - 0-10)
template['source'] = "How novel is the movie_{} for user_{}? (0 being not novel, 10 being very novel)"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "6-3"
task_subgroup_1["6-3"] = template
template = {}

# Template 6-4 (Novelty - Three choices)
template['source'] = "For user_{}, is movie_{} a film they likely haven't seen or heard of before? (0 being lowest and 10 being highest)"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "6-4"
task_subgroup_1["6-4"] = template
template = {}

# Template 6-5 (Serendipity - 0-10)
template['source'] = "How serendipitous (unexpectedly good) is the movie_{} for user_{}? (0 being not, 10 being highly serendipitous)"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "6-5"
task_subgroup_1["6-5"] = template
template = {}

# Template 6-6 (Serendipity - Three choices)
template['source'] = "For user_{}, is movie_{} an unexpected film that they would surprisingly enjoy? (0 being lowest and 10 being highest)"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "6-6"
task_subgroup_1["6-6"] = template
template = {}

all_tasks['diversity'] = task_subgroup_6

# Add other task families if needed, but for rating prediction, this is the primary one.
# For P5's multitask nature, you might need to define minimal templates for other task types
# even if you don't have rich data for them, or ensure the dataloader handles their absence.

# --- Minimal Sequential Prompts (if P5 framework strictly requires it) ---
task_subgroup_2 = {}
template = {}
template['source'] = "Given the viewing history of user_{}: {}, predict the next possible movie."
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'view_history'] # view_history will be a string of item_ids
template['target_argc'] = 1
template['target_argv'] = ['next_item_id']
template['id'] = "2-1-ml1m" # Unique ID for ML1M
task_subgroup_2["2-1-ml1m"] = template
all_tasks['sequential'] = task_subgroup_2
template = {}

# --- Minimal Explanation Prompts (empty/placeholder if not used) ---
task_subgroup_3 = {}
template = {}
template['source'] = "Why might user_{} have rated movie_{} with {} stars?" # Placeholder
template['target'] = "{}" # Placeholder: "No explanation available."
template['task'] = "explanation"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'item_id', 'star_rating']
template['target_argc'] = 1
template['target_argv'] = ['explanation_text']
template['id'] = "3-1-ml1m"
task_subgroup_3["3-1-ml1m"] = template
all_tasks['explanation'] = task_subgroup_3
template = {}

# --- Minimal Review Prompts (empty/placeholder if not used) ---
task_subgroup_4 = {}
template = {}
template['source'] = "What is the rating for a review of movie_{} by user_{}?" # Placeholder
template['target'] = "{}"
template['task'] = "review" # This task in P5 often means predicting rating from review text
template['source_argc'] = 2
template['source_argv'] = ['item_id', 'user_id'] # Simplified for ML1M as no review text
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "4-1-ml1m"
task_subgroup_4["4-1-ml1m"] = template
all_tasks['review'] = task_subgroup_4
template = {}

# --- Minimal Traditional Recommendation Prompts (if P5 framework strictly requires it) ---
task_subgroup_5 = {}
template = {}
template['source'] = "Will user_{} like movie_{}?"
template['target'] = "{}" # yes/no
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "5-1-ml1m"
task_subgroup_5["5-1-ml1m"] = template
all_tasks['traditional'] = task_subgroup_5
template = {}

# Note: For a rating-prediction focused setup, you might primarily use 'rating' tasks.
# The other minimal tasks are included for broader compatibility if the P5 data loader or training loop
# expects definitions for all task types listed in args.losses.