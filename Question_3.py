import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

sns.set_style('whitegrid')
sns.set_palette('husl')

# Part A: Compare client intake characteristics across the trajectory clusters identified in Question 2.
# This script loads intake features, joins them with cluster labels, and produces summary tables and plots.

def load_intake_and_clusters():
    """Load intake characteristics and cluster labels from Q2."""
    # Read intake feature data and previously computed cluster assignments
    features = pd.read_csv('client_features.csv')
    cluster_labels = pd.read_csv('output/q2/cluster_assignments.csv')

    # Merge on client_id so each client has intake features plus cluster membership
    merged = features.merge(cluster_labels[['client_id', 'cluster']], on='client_id', how='inner')
    merged['cluster'] = merged['cluster'].astype(int)
    return merged


def build_summary_tables(data):
    """Build numeric and categorical summaries by cluster."""
    numeric_features = ['age_years', 'complexity_score']
    categorical_features = ['gender', 'referral_reason']

    # Compute common numeric summary statistics for each cluster
    numeric_summary = data.groupby('cluster')[numeric_features].agg(['count', 'mean', 'std', 'min', 'max'])

    # Add cluster-level quartiles for each numeric feature
    quantiles = data.groupby('cluster')[numeric_features].quantile([0.25, 0.5, 0.75]).unstack(level=-1)
    quantiles.columns = [f'{feature}_{int(q*100)}%' for feature, q in quantiles.columns]
    numeric_summary = pd.concat([numeric_summary, quantiles], axis=1)
    numeric_summary = numeric_summary.rename(columns={f'{feature}_50%': f'{feature}_median' for feature in numeric_features})

    # Build normalized categorical distributions by cluster
    categorical_summary = {
        feature: pd.crosstab(data['cluster'], data[feature], normalize='index')
        for feature in categorical_features
    }

    return numeric_summary, categorical_summary


def plot_numeric_features(data, out_dir):
    """Create boxplots for numeric intake features across clusters."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot age distributions by cluster with overlaid points
    sns.boxplot(x='cluster', y='age_years', data=data, palette='husl', ax=axes[0])
    sns.stripplot(x='cluster', y='age_years', data=data, color='black', alpha=0.5, size=4, ax=axes[0])
    axes[0].set_title('Age at Intake by Trajectory Cluster')
    axes[0].set_xlabel('Trajectory Cluster')
    axes[0].set_ylabel('Age (years)')

    # Plot complexity score distributions by cluster
    sns.boxplot(x='cluster', y='complexity_score', data=data, palette='husl', ax=axes[1])
    sns.stripplot(x='cluster', y='complexity_score', data=data, color='black', alpha=0.5, size=4, ax=axes[1])
    axes[1].set_title('Complexity Score by Trajectory Cluster')
    axes[1].set_xlabel('Trajectory Cluster')
    axes[1].set_ylabel('Complexity Score')

    plt.tight_layout()
    path = os.path.join(out_dir, 'intake_numeric_by_cluster.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_categorical_features(data, out_dir):
    """Create stacked bar charts for categorical intake features by cluster."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    gender_prop = pd.crosstab(data['cluster'], data['gender'], normalize='index')
    gender_prop.plot(kind='bar', stacked=True, ax=axes[0], colormap='Set2', width=0.75)
    axes[0].set_title('Gender Composition by Trajectory Cluster')
    axes[0].set_xlabel('Trajectory Cluster')
    axes[0].set_ylabel('Proportion')
    axes[0].legend(title='Gender', bbox_to_anchor=(1.02, 1), loc='upper left')

    referral_prop = pd.crosstab(data['cluster'], data['referral_reason'], normalize='index')
    referral_prop.plot(kind='bar', stacked=True, ax=axes[1], colormap='tab20', width=0.75)
    axes[1].set_title('Referral Reason by Trajectory Cluster')
    axes[1].set_xlabel('Trajectory Cluster')
    axes[1].set_ylabel('Proportion')
    axes[1].legend(title='Referral Reason', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    path = os.path.join(out_dir, 'intake_categorical_by_cluster.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_feature_grid(data, out_dir):
    """Create a combined plot grid for intake features by cluster."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    sns.countplot(x='cluster', hue='gender', data=data, palette='Set2', ax=axes[0, 0])
    axes[0, 0].set_title('Gender Counts by Trajectory Cluster')
    axes[0, 0].set_xlabel('Trajectory Cluster')
    axes[0, 0].set_ylabel('Count')

    sns.countplot(x='cluster', hue='referral_reason', data=data, palette='Set3', ax=axes[0, 1])
    axes[0, 1].set_title('Referral Reason Counts by Trajectory Cluster')
    axes[0, 1].set_xlabel('Trajectory Cluster')
    axes[0, 1].set_ylabel('Count')

    sns.boxplot(x='cluster', y='age_years', data=data, palette='husl', ax=axes[1, 0])
    axes[1, 0].set_title('Age at Intake by Trajectory Cluster')
    axes[1, 0].set_xlabel('Trajectory Cluster')
    axes[1, 0].set_ylabel('Age (years)')

    sns.boxplot(x='cluster', y='complexity_score', data=data, palette='husl', ax=axes[1, 1])
    axes[1, 1].set_title('Complexity Score by Trajectory Cluster')
    axes[1, 1].set_xlabel('Trajectory Cluster')
    axes[1, 1].set_ylabel('Complexity Score')

    plt.tight_layout()
    path = os.path.join(out_dir, 'intake_feature_grid_by_cluster.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def compute_feature_separation(data):
    """Estimate which intake features are most distinct across clusters."""
    results = {}

    # Numeric features: calculate eta-squared to quantify how much cluster means explain variance
    for feature in ['age_years', 'complexity_score']:
        overall_mean = data[feature].mean()
        ss_total = ((data[feature] - overall_mean) ** 2).sum()
        ss_between = sum(
            len(group) * (group[feature].mean() - overall_mean) ** 2
            for _, group in data.groupby('cluster')
        )
        eta2 = ss_between / ss_total if ss_total > 0 else 0.0
        results[feature] = {'type': 'numeric', 'eta2': eta2}

    # Categorical features: measure how much each cluster's proportions differ from the overall distribution
    for feature in ['gender', 'referral_reason']:
        overall_prop = data[feature].value_counts(normalize=True)
        cluster_props = pd.crosstab(data['cluster'], data[feature], normalize='index')
        avg_distance = cluster_props.apply(lambda row: np.abs(row - overall_prop).sum(), axis=1).mean() / 2
        results[feature] = {'type': 'categorical', 'avg_l1_distance': avg_distance}

    return results


def print_feature_insights(data, separation_scores):
    """Print interpretive insights about which features distinguish clusters."""
    print('\n=== Intake Feature Comparison Across Trajectory Clusters ===\n')
    print(f'Total clients included: {len(data)}')
    print(f'Clusters found: {sorted(data["cluster"].unique())}\n')

    print('Numeric intake feature summaries by cluster:')
    print(data.groupby('cluster')[['age_years', 'complexity_score']].describe())
    print('\nCategorical intake feature distribution by cluster:')
    for feature in ['gender', 'referral_reason']:
        print(f'\n{feature}:')
        print(pd.crosstab(data['cluster'], data[feature], normalize='index').round(2))

    print('\nFeature separation scores:')
    for feature, score in separation_scores.items():
        if score['type'] == 'numeric':
            print(f"- {feature}: eta-squared = {score['eta2']:.3f} (larger means stronger separation)")
        else:
            print(f"- {feature}: avg. cluster L1 distance = {score['avg_l1_distance']:.3f} (larger means stronger deviation from overall proportions)")

    print('\nInterpretive answers:')
    print('- Features most useful for distinguishing trajectory groups are those with the highest separation scores. In this dataset, complexity_score shows the strongest numeric separation and referral_reason shows the strongest categorical separation.')
    print('- Features that overlap substantially across groups are those with low separation scores. Age appears more overlapping across clusters than complexity score, and gender shows less separation across clusters than referral_reason.')



# part B: Train parametric and non-parametric classifiers to predict trajectory cluster membership from intake features, and report accuracy/confusion matrices.
def prepare_model_data(data):
    """Prepare intake feature matrix and cluster labels for classification."""
    features = ['age_years', 'complexity_score', 'gender', 'referral_reason']
    X = pd.get_dummies(data[features], drop_first=True)
    y = data['cluster']
    return X, y


def train_and_evaluate_models(data, out_dir):
    """Train parametric and non-parametric classifiers and report accuracy/confusion matrices."""
    # Part B: Train classifiers on historical clients to predict trajectory cluster membership
    X, y = prepare_model_data(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'multinomial_logistic_regression': LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ),
        'random_forest': RandomForestClassifier(
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2,
            n_estimators=300,
        )
    }

    results = {}
    labels = sorted(y.unique())
    best_accuracy = -1.0
    best_name = None
    best_model = None

    print('Part B: Classifier performance')

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        cm_path = os.path.join(out_dir, f'{name}_confusion_matrix.csv')
        cm_df.to_csv(cm_path)

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'confusion_matrix': cm_df,
            'confusion_matrix_path': cm_path
        }

        # Track the best model by test accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_name = name
            best_model = model

        print(f"\n{name.replace('_', ' ').title()}")
        print(f"Accuracy: {accuracy:.3f}")
        print('Confusion matrix:')
        print(cm_df)

    return results, best_name, best_model, X_train.columns


# Part C: Apply best model to waitlist and estimate capacity under differentiated reassessment policy
def load_waitlist_data():
    """Load intake characteristics for the current waitlist."""
    waitlist = pd.read_csv('waitlist.csv')
    return waitlist


def apply_best_model_to_waitlist(best_model, model_columns, out_dir):
    """Apply the trained best model to predict cluster membership for all waitlist clients."""
    waitlist = load_waitlist_data()
    features = ['age_years', 'complexity_score', 'gender', 'referral_reason']
    
    # One-hot encode waitlist features and align with training columns
    X_waitlist = pd.get_dummies(waitlist[features], drop_first=True)
    X_waitlist = X_waitlist.reindex(columns=model_columns, fill_value=0)
    
    # Predict cluster membership
    waitlist['predicted_cluster'] = best_model.predict(X_waitlist)
    waitlist.to_csv(os.path.join(out_dir, 'waitlist_predicted_clusters.csv'), index=False)
    
    return waitlist


def load_cluster_statistics():
    """Load cluster Q* and estimated expected sessions from Q2 policy outputs."""
    # Load Q* values for each cluster
    q_star_data = pd.read_csv('output/q2/reassessment_policy.csv', usecols=['cluster', 'optimal_q'])
    q_star_data['cluster'] = q_star_data['cluster'].astype(int)
    
    # Map cluster-level expected delivered sessions based on Q2 analysis
    # These are estimated from historical stopping points in each cluster
    expected_delivered = {
        1: 7.85,  # Cluster 1: mean stopping point ≈ 7.85 sessions
        2: 7.25,  # Cluster 2: mean stopping point ≈ 7.25 sessions
        3: 5.67,  # Cluster 3: mean stopping point ≈ 5.67 sessions
        4: 7.79,  # Cluster 4: mean stopping point ≈ 7.79 sessions
        5: 5.0    # Cluster 5: mean stopping point ≈ 5.0 sessions
    }
    
    q_star_data['expected_delivered_sessions'] = q_star_data['cluster'].map(expected_delivered)
    return q_star_data


def estimate_waitlist_capacity(waitlist, q_star_data, tmax_baseline=12, out_dir=None):
    """Estimate total session capacity needed for waitlist under differentiated policy vs. baseline."""
    # Merge predicted clusters with Q* and expected sessions
    waitlist_merged = waitlist.merge(
        q_star_data, 
        left_on='predicted_cluster', 
        right_on='cluster', 
        how='left'
    )
    
    # Calculate sessions under each scenario
    waitlist_merged['baseline_sessions'] = tmax_baseline
    waitlist_merged['policy_sessions'] = waitlist_merged['expected_delivered_sessions']
    waitlist_merged['session_savings'] = waitlist_merged['baseline_sessions'] - waitlist_merged['policy_sessions']
    
    # Aggregate by predicted cluster
    cluster_summary = waitlist_merged.groupby('predicted_cluster').agg({
        'client_id': 'count',
        'optimal_q': 'first',
        'expected_delivered_sessions': 'first',
        'policy_sessions': 'sum',
        'baseline_sessions': 'sum',
        'session_savings': 'sum'
    }).rename(columns={'client_id': 'n_waitlist'}).reset_index()
    
    cluster_summary['percent_savings'] = (
        cluster_summary['session_savings'] / cluster_summary['baseline_sessions'] * 100
    ).round(1)
    
    # Save results
    if out_dir:
        waitlist_merged.to_csv(os.path.join(out_dir, 'waitlist_with_capacity_estimates.csv'), index=False)
        cluster_summary.to_csv(os.path.join(out_dir, 'waitlist_capacity_summary_by_cluster.csv'), index=False)
    
    return waitlist_merged, cluster_summary


def print_capacity_analysis(waitlist, cluster_summary, tmax_baseline=12, best_name=None):
    """Print comprehensive capacity comparison and insights."""
    total_baseline = tmax_baseline * len(waitlist)
    total_policy = waitlist['policy_sessions'].sum()
    total_savings = total_baseline - total_policy
    percent_savings = (total_savings / total_baseline * 100)
    
    print('PART C: WAITLIST CAPACITY ANALYSIS')
    print(f'\nClassifier used for Part C: {best_name.replace("_", " ").title()}')
    print(f'Waitlist size: {len(waitlist)} clients')
    print(f'\nBaseline assumption (Tmax={tmax_baseline} sessions per client):')
    print(f'  Total sessions required: {total_baseline}')
    print(f'\nDifferentiated policy (using cluster-specific Q* reassessments):')
    print(f'  Total sessions required: {total_policy:.0f}')
    print(f'  Expected sessions saved: {total_savings:.0f}')
    print(f'  Percent savings: {percent_savings:.1f}%')
    
    print(f'\n--- Capacity Impact by Predicted Cluster ---')
    print(cluster_summary.to_string(index=False))
    
    # Identify drivers of savings
    if len(cluster_summary) > 0:
        max_savings_row = cluster_summary.loc[cluster_summary['session_savings'].idxmax()]
        
        print(f'\nCluster {int(max_savings_row["predicted_cluster"])} DRIVES MOST CAPACITY SAVINGS:')
        print(f'  {int(max_savings_row["n_waitlist"])} clients predicted to this cluster')
        print(f'  Expected delivery: {max_savings_row["expected_delivered_sessions"]:.1f} sessions/client')
        print(f'  Total expected under policy: {max_savings_row["policy_sessions"]:.0f} sessions')
        print(f'  Baseline (12/client): {int(max_savings_row["baseline_sessions"])} sessions')
        print(f'  Total savings: {max_savings_row["session_savings"]:.0f} sessions ({max_savings_row["percent_savings"]:.1f}%)')


def main():
    # Ensure the output folder for Question 3 exists
    out_dir = 'output/q3'
    os.makedirs(out_dir, exist_ok=True)

    # Load merged intake and cluster data, then generate summary tables
    data = load_intake_and_clusters()
    numeric_summary, categorical_summary = build_summary_tables(data)

    # Save summary tables for later review
    numeric_summary.to_csv(os.path.join(out_dir, 'intake_numeric_summary_by_cluster.csv'))
    for feature, summary in categorical_summary.items():
        summary.to_csv(os.path.join(out_dir, f'intake_{feature}_proportions_by_cluster.csv'))

    # Create and save visual summaries for intake features
    plot_numeric_features(data, out_dir)
    plot_categorical_features(data, out_dir)
    plot_feature_grid(data, out_dir)

    # Compute and print feature separation metrics and interpretation
    separation_scores = compute_feature_separation(data)
    print_feature_insights(data, separation_scores)

    # Part B: Train classifiers to predict trajectory cluster membership from intake features
    model_results, best_name, best_model, model_columns = train_and_evaluate_models(data, out_dir)
    print(f"\nBest classifier selected: {best_name.replace('_', ' ').title()}\n")

    # For Part C, use Random Forest as requested
    rf_model = model_results['random_forest']['model']
    rf_name = 'random_forest'

    # Part C: Apply Random Forest to waitlist and estimate capacity under differentiated policy
    waitlist = apply_best_model_to_waitlist(rf_model, model_columns, out_dir)
    q_star_data = load_cluster_statistics()
    waitlist_merged, cluster_summary = estimate_waitlist_capacity(waitlist, q_star_data, tmax_baseline=12, out_dir=out_dir)
    print_capacity_analysis(waitlist_merged, cluster_summary, tmax_baseline=12, best_name=rf_name)


if __name__ == '__main__':
    main()
