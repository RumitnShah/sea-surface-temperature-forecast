if hasattr(model, "feature_importances_"):
    #     importances = model.feature_importances_

    #     imp_df = pd.DataFrame({
    #         "feature": FEATURE_COLS,
    #         "importance": importances
    #     }).sort_values(by="importance", ascending=False)

    #     print("\nTop Features:")
    #     print(imp_df.head(5))