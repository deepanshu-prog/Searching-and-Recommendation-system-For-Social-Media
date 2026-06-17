#this code generate Dataset of the user along with thire interaction score 
import pandas as pd
import random


## this file is used to generate dataset in csv format for Code.py file it is for representation purpose only 
## this code always generate minimum 2 user with same name but with different user_id just like in real life cases
# 1. Setup Data
indian_names = ["Aarav", "Vihaan", "Aditya", "Sai", "Arjun", "Diya", "Ananya", "Riya", "Isha", "Kavya"]
locations = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune", "Kolkata"]

# 2. Create Users
# change num_user according to you choice 
num_users = 5
users = []
duplicate_name = random.choice(indian_names)

# User 101 and 102 have the same name to show that how it helps in finding the best serach/recommendation 
users.append({"user_id": 101, "username": duplicate_name, "location": random.choice(locations)})
users.append({"user_id": 102, "username": duplicate_name, "location": random.choice(locations)})

# Create remaining users
for i in range(103, 100 + num_users + 1):
    users.append({"user_id": i, "username": random.choice(indian_names), "location": random.choice(locations)})

# 3. Generate Connections (Edges)
weights = {"likes": 0.1, "comments": 0.2, "messages": 0.4, "story_views": 0.3}
max_values = {"likes": 20, "comments": 10, "messages": 25, "story_views": 15}
records = []

for u in users:
    # Each user connects to random friends
    friend_candidates = [f for f in users if f["user_id"] != u["user_id"]]
    friends = random.sample(friend_candidates, k=random.randint(1, len(friend_candidates)))

    for f in friends:
        # Random engagement stats
        likes = random.randint(0, max_values["likes"])
        comments = random.randint(0, max_values["comments"])
        messages = random.randint(0, max_values["messages"])
        story_views = random.randint(0, max_values["story_views"])

        # Calculate Score
        score = (
            weights["likes"] * (likes / max_values["likes"]) +
            weights["comments"] * (comments / max_values["comments"]) +
            weights["messages"] * (messages / max_values["messages"]) +
            weights["story_views"] * (story_views / max_values["story_views"])
        )
        
        # Boost if locations match
        if u["location"] == f["location"]:
            score = min(score * 1.1, 1.0)

        records.append({
            "user_id": u["user_id"],
            "username": u["username"],
            "friend_id": f["user_id"],
            "friend_name": f["username"],
            "location": u["location"],
            "friend_location": f["location"],
            "likes": likes,
            "comments": comments,
            "messages": messages,
            "story_views": story_views,
            "engagement_score": round(score, 3),
            "edge_weight": round(1 - score, 3)  # Weight for SSSP
        })

# 4. Save to CSV
df = pd.DataFrame(records)
df.to_csv("social_engagement.csv", index=False)
print("✅ 'social_engagement.csv' created successfully!")