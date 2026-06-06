import pandas as pd
import json
import random
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# ==============================================================================
# 1. PYDANTIC SCHEMAS FOR STRUCTURED LLM OUTPUTS
# ==============================================================================

class MotivationOutput(BaseModel):
    pred_FOMO: int = Field(description="Predicted Fear of Missing Out score (e.g., 10-40)")
    motivation_analysis: str = Field(description="Clinical analysis of why the user posts (validation, support, etc.)")

class PersonalityOutput(BaseModel):
    pred_BFI_N: int = Field(description="Predicted Neuroticism score (Big Five)")
    pred_Lonely: int = Field(description="Predicted UCLA Loneliness score (10-40)")
    pred_Brooding: int = Field(description="Predicted Brooding/Rumination score (5-20)")
    pred_Worry: int = Field(description="Predicted PSWQ Worry score (16-80)")
    pred_SWL: int = Field(description="Predicted Satisfaction With Life score (5-35)")
    personality_analysis: str = Field(description="Analysis of traits and psychosocial distress based on motivation")

class ClinicalOutput(BaseModel):
    pred_PHQ9_1: int = Field(description="PHQ-9 Item 1: Anhedonia (0-3)")
    pred_PHQ9_2: int = Field(description="PHQ-9 Item 2: Depressed mood (0-3)")
    pred_PHQ9_3: int = Field(description="PHQ-9 Item 3: Sleep issues (0-3)")
    pred_PHQ9_4: int = Field(description="PHQ-9 Item 4: Fatigue (0-3)")
    pred_PHQ9_5: int = Field(description="PHQ-9 Item 5: Appetite changes (0-3)")
    pred_PHQ9_6: int = Field(description="PHQ-9 Item 6: Guilt/Worthlessness (0-3)")
    pred_PHQ9_7: int = Field(description="PHQ-9 Item 7: Concentration issues (0-3)")
    pred_PHQ9_8: int = Field(description="PHQ-9 Item 8: Psychomotor agitation/retardation (0-3)")
    pred_PHQ9_9: int = Field(description="PHQ-9 Item 9: Suicidal ideation (0-3)")
    pred_MDD: int = Field(description="Binary indicator for Major Depressive Disorder (0 = No, 1 = Yes)")
    clinical_analysis: str = Field(description="Psychiatric formulation of mood disorders")

class RiskOutput(BaseModel):
    pred_SD: int = Field(description="Predicted Suicide Risk Severity on a scale matching the dataset (e.g., 0-6)")
    risk_analysis: str = Field(description="Final C-SSRS based risk assessment justification")

# ==============================================================================
# 2. HELPER FUNCTIONS FOR API CALLS & CONTEXT BUILDING
# ==============================================================================

def call_llm_agent(system_prompt: str, user_context: str, response_schema: BaseModel) -> Any:
    """
    Mock function representing an API call to an LLM (e.g., GPT-4o) using Structured Outputs.
    In a real scenario, use: client.beta.chat.completions.parse(...)
    """
    # Example integration with OpenAI Python SDK:
    # response = client.beta.chat.completions.parse(
    #     model="gpt-4o",
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_context}
    #     ],
    #     response_format=response_schema
    # )
    # return response.choices[0].message.parsed
    pass

def build_user_context(row: pd.Series, user_posts: List[str]) -> str:
    """
    Concatenates metadata and chronological posts into a single context block.
    """
    context = f"""
    [RAW USER PROFILE]
    Platform Friends: {row.get('FriendCount', 'N/A')}
    Total Posts (Year): {row.get('status_posts', 'N/A')}
    Age: {row.get('age', 'N/A')}
    Gender: {'Female' if row.get('female') == 1 else 'Male'}
    Annual Income: ${row.get('inc_num', 'N/A')}
    
    [CHRONOLOGICAL POSTS]
    """
    for i, post in enumerate(user_posts):
        context += f"(Post {i+1}): \"{post}\"\n"
    
    return context

# ==============================================================================
# 3. MAIN PIPELINE EXECUTION
# ==============================================================================

def run_evaluation_pipeline(features_csv_path: str):
    # Load dataset
    df = pd.read_csv(features_csv_path)
    
    # Filter valid users (e.g., relevant groups and sufficient posts)
    valid_users = df[(df['status_posts'] > 9) & (df['grp'].isin([0, 1]))].copy()
    
    # Shuffle and split 10% Dev (Few-Shot) / 90% Test
    users_list = valid_users.to_dict('records')
    random.shuffle(users_list)
    split_index = int(len(users_list) * 0.1)
    
    dev_set = users_list[:split_index]   # For crafting the few-shot prompts
    test_set = users_list[split_index:]  # For blind evaluation
    
    print(f"Total valid users: {len(users_list)} | Dev Set (Few-Shot): {len(dev_set)} | Test Set: {len(test_set)}")
    
    results = []
    
    # Iterate through the Test Set
    for user_row in test_set:
        user_id = user_row['UserId']
        print(f"Processing User: {user_id}...")
        
        # MOCK: Fetch actual posts from your database for this user
        # user_posts = fetch_posts_from_db(user_id)
        user_posts = ["I feel so alone today", "Nobody ever invites me anywhere", "Back pain is killing me..."] 
        
        # 1. Build initial context
        accumulated_context = build_user_context(user_row, user_posts)
        
        # ----------------------------------------------------------------------
        # AGENT 1: Social Media Motivation
        # ----------------------------------------------------------------------
        sys_prompt_1 = "You are an expert analyzing social media behavior. Assess the user's motivation and FOMO."
        # Inject few-shot examples from dev_set into sys_prompt_1 here
        
        motivation_res = call_llm_agent(sys_prompt_1, accumulated_context, MotivationOutput)
        
        # Append Agent 1 findings to context
        accumulated_context += f"\n\n[AGENT 1 - MOTIVATION ANALYSIS]\n{motivation_res.json()}"
        
        # ----------------------------------------------------------------------
        # AGENT 2: Personality & Psychosocial Distress
        # ----------------------------------------------------------------------
        sys_prompt_2 = "You are a Psychologist. Based on the raw data and motivation analysis, evaluate Big Five traits and distress."
        
        personality_res = call_llm_agent(sys_prompt_2, accumulated_context, PersonalityOutput)
        accumulated_context += f"\n\n[AGENT 2 - PERSONALITY & DISTRESS]\n{personality_res.json()}"
        
        # ----------------------------------------------------------------------
        # AGENT 3: Clinical Diagnostic (DSM-5 / PHQ-9)
        # ----------------------------------------------------------------------
        sys_prompt_3 = "You are a Psychiatrist. Translate the psychosocial profile and raw posts into PHQ-9 criteria and MDD diagnosis."
        
        clinical_res = call_llm_agent(sys_prompt_3, accumulated_context, ClinicalOutput)
        accumulated_context += f"\n\n[AGENT 3 - CLINICAL DIAGNOSIS]\n{clinical_res.json()}"
        
        # ----------------------------------------------------------------------
        # AGENT 4: Suicide Risk Assessment (C-SSRS)
        # ----------------------------------------------------------------------
        sys_prompt_4 = "You are a Risk Assessor. Use the accumulated clinical file to predict the final Suicide Risk Severity (SD)."
        
        risk_res = call_llm_agent(sys_prompt_4, accumulated_context, RiskOutput)
        
        # ----------------------------------------------------------------------
        # SAVE PREDICTIONS VS GROUND TRUTH
        # ----------------------------------------------------------------------
        result_entry = {
            "UserId": user_id,
            # Ground Truths
            "true_FOMO": user_row.get('FOMO'),
            "true_BFI_N": user_row.get('BFI_N'),
            "true_Lonely": user_row.get('Lonely'),
            "true_MDD": user_row.get('MDD'),
            "true_SD": user_row.get('SD'),
            
            # Predictions
            "pred_FOMO": motivation_res.pred_FOMO,
            "pred_BFI_N": personality_res.pred_BFI_N,
            "pred_Lonely": personality_res.pred_Lonely,
            "pred_MDD": clinical_res.pred_MDD,
            "pred_SD": risk_res.pred_SD,
            
            # Analyses (for qualitative error review)
            "final_risk_analysis": risk_res.risk_analysis
        }
        results.append(result_entry)

    # Convert to DataFrame for metric calculations (Pearson, F1, MSE/AUC-ROC)
    results_df = pd.DataFrame(results)
    results_df.to_csv('pipeline_evaluation_results.csv', index=False)
    print("\nPipeline execution complete. Results saved to 'pipeline_evaluation_results.csv'.")

# if __name__ == "__main__":
#     run_evaluation_pipeline('features_exp(Sheet1).csv')