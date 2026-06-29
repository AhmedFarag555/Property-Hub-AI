from smart_recommender import SmartRecommender

r = SmartRecommender()

results = r.recommend(user_id=1)

for i in results:
    print(i["title"])