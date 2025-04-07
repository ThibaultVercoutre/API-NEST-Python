import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy as np

model = "deepseek"

def analyze_accuracy():
    """
    Analyse la précision des classifications du LLM en comparant avec les labels POI
    """
    conn = sqlite3.connect('email_classifications.db')
    # Chargement des résultats
    try:
        results = pd.read_sql_query(f"SELECT * FROM results WHERE model = '{model}'", conn)
        print(f"Nombre total d'emails analysés: {len(results)}")
        
        # Création d'une colonne pour les bonnes prédictions
        results['correct_prediction'] = (
            ((results['poi_present'] == 1) & 
            (results['classification'] == 'NONOK')) |
            ((results['poi_present'] == 0) & 
            (results['classification'] == 'OK'))
        )
        
        # Calcul des métriques
        total_samples = len(results)
        correct_predictions = results['correct_prediction'].sum()
        accuracy = (correct_predictions / total_samples) * 100
        
        # Distribution des classifications
        classification_dist = results['classification'].value_counts()
        
        # Affichage des résultats
        print("\nRésultats de l'analyse:")
        print(f"Nombre total d'échantillons: {total_samples}")
        print(f"Nombre de prédictions correctes: {correct_predictions}")
        print(f"Précision globale: {accuracy:.2f}%")
        
        print("\nDistribution des classifications:")
        for class_name, count in classification_dist.items():
            percentage = (count / total_samples) * 100
            print(f"{class_name}: {count} ({percentage:.2f}%)")
            
        # Analyse détaillée par type de classification
        print("\nAnalyse détaillée:")
        for classification in ['NONOK', 'OK']:
            subset = results[results['classification'] == classification]
            if len(subset) > 0:
                correct = subset['correct_prediction'].sum()
                accuracy = (correct / len(subset)) * 100
                print(f"\nPour la classification {classification}:")
                print(f"Nombre total: {len(subset)}")
                print(f"Prédictions correctes: {correct}")
                print(f"Précision: {accuracy:.2f}%")

        # Analyse des vrais positifs et vrais négatifs
        print("\nAnalyse détaillée des POI:")
        
        # Pour les emails contenant réellement des POI (poi_present = 1)
        poi_emails = results[results['poi_present'] == 1]
        if len(poi_emails) > 0:
            correct_poi = poi_emails[poi_emails['classification'] == 'NONOK']
            print(f"\nEmails contenant des POI (poi_present = 1):")
            print(f"Nombre total: {len(poi_emails)}")
            print(f"Correctement identifiés comme NONOK: {len(correct_poi)}")
            print(f"Précision: {(len(correct_poi) / len(poi_emails) * 100):.2f}%")
        
        # Pour les emails ne contenant pas de POI (poi_present = 0)
        non_poi_emails = results[results['poi_present'] == 0]
        if len(non_poi_emails) > 0:
            correct_non_poi = non_poi_emails[non_poi_emails['classification'] == 'OK']
            print(f"\nEmails sans POI (poi_present = 0):")
            print(f"Nombre total: {len(non_poi_emails)}")
            print(f"Correctement identifiés comme OK: {len(correct_non_poi)}")
            print(f"Précision: {(len(correct_non_poi) / len(non_poi_emails) * 100):.2f}%")
        
    except Exception as e:
        print(f"Une erreur est survenue: {str(e)}")

def remove_ok_classifications():
    """
    Supprime toutes les lignes où la classification est 'OK' du fichier CSV.
    """
    try:
        results = pd.read_csv('llm_' + model + '_test_results.csv')
        initial_count = len(results)
        
        # Suppression des lignes où la classification est 'OK'
        results = results[results['llm_classification'] != 'OK']
        final_count = len(results)
        
        # Sauvegarde du fichier CSV mis à jour
        results.to_csv('llm_' + model + '_test_results.csv', index=False)
        
        print(f"Nombre de lignes supprimées: {initial_count - final_count}")
        print(f"Nombre de lignes restantes: {final_count}")
        
    except FileNotFoundError:
        print("Erreur: Le fichier 'llm_mistral_test_results.csv' n'a pas été trouvé.")
    except Exception as e:
        print(f"Une erreur est survenue: {str(e)}")

def analyze_accuracy_combined():
    """
    Analyse la précision des classifications du LLM en combinant les résultats de plusieurs modèles.
    """
    models = ['phi', 'phi3', 'mistral', 'deepseek', 'phishing']
    conn = sqlite3.connect('email_classifications.db')
    
    try:
        combined_results = pd.DataFrame()
        
        for model in models:
            results = pd.read_sql_query(f"SELECT * FROM results WHERE model = '{model}'", conn)
            results['model'] = model
            combined_results = pd.concat([combined_results, results])
        
        combined_results['final_classification'] = combined_results.groupby('hash')['classification'].transform(
            lambda x: 'OK' if (x == 'OK').sum() > (x == 'NONOK').sum() else ('NONOK' if (x == 'NONOK').sum() > (x == 'OK').sum() else 'UNKNOWN')
        )
        
        combined_results['correct_prediction'] = (
            ((combined_results['poi_present'] == 1) & 
            (combined_results['final_classification'] == 'NONOK')) |
            ((combined_results['poi_present'] == 0) & 
            (combined_results['final_classification'] == 'OK'))
        )
        
        total_samples = len(combined_results)
        correct_predictions = combined_results['correct_prediction'].sum()
        accuracy = (correct_predictions / total_samples) * 100
        
        classification_dist = combined_results['final_classification'].value_counts()
        
        print("\nRésultats de l'analyse combinée:")
        print(f"Nombre total d'échantillons: {total_samples}")
        print(f"Nombre de prédictions correctes: {correct_predictions}")
        print(f"Précision globale: {accuracy:.2f}%")
        
        print("\nDistribution des classifications:")
        for class_name, count in classification_dist.items():
            percentage = (count / total_samples) * 100
            print(f"{class_name}: {count} ({percentage:.2f}%)")
        
        print("\nAnalyse détaillée:")
        for classification in ['NONOK', 'OK', 'UNKNOWN']:
            subset = combined_results[combined_results['final_classification'] == classification]
            if len(subset) > 0:
                correct = subset['correct_prediction'].sum()
                accuracy = (correct / len(subset)) * 100
                print(f"\nPour la classification {classification}:")
                print(f"Nombre total: {len(subset)}")
                print(f"Prédictions correctes: {correct}")
                print(f"Précision: {accuracy:.2f}%")
        
        print("\nAnalyse détaillée des POI:")
        
        poi_emails = combined_results[combined_results['poi_present'] == 1]
        if len(poi_emails) > 0:
            correct_poi = poi_emails[poi_emails['final_classification'] == 'NONOK']
            print(f"\nEmails contenant des POI (poi_present = 1):")
            print(f"Nombre total: {len(poi_emails)}")
            print(f"Correctement identifiés comme NONOK: {len(correct_poi)}")
            print(f"Précision: {(len(correct_poi) / len(poi_emails) * 100):.2f}%")
        
        non_poi_emails = combined_results[combined_results['poi_present'] == 0]
        if len(non_poi_emails) > 0:
            correct_non_poi = non_poi_emails[non_poi_emails['final_classification'] == 'OK']
            print(f"\nEmails sans POI (poi_present = 0):")
            print(f"Nombre total: {len(non_poi_emails)}")
            print(f"Correctement identifiés comme OK: {len(correct_non_poi)}")
            print(f"Précision: {(len(correct_non_poi) / len(non_poi_emails) * 100):.2f}%")
        
    except Exception as e:
        print(f"Une erreur est survenue: {str(e)}")

def analyze_accuracy_combined_2():
    """
    Analyse la précision des classifications du LLM en combinant les résultats de plusieurs modèles.
    Version 2 avec analyse plus détaillée.
    """
    models = ['phi', 'phi3', 'mistral', 'deepseek', 'phishing']
    conn = sqlite3.connect('email_classifications.db')
    
    try:
        # Récupération des données pour chaque modèle
        combined_results = pd.DataFrame()
        
        for model in models:
            results = pd.read_sql_query(f"SELECT * FROM results WHERE model = '{model}'", conn)
            results['model'] = model
            combined_results = pd.concat([combined_results, results])

        # Grouper par hash et compter le nombre de modèles par hash
        hash_counts = combined_results.groupby('hash').size()
        valid_hashes = hash_counts[hash_counts == 4].index
        
        # Filtrer pour ne garder que les hashs avec les 4 modèles
        filtered_results = combined_results[combined_results['hash'].isin(valid_hashes)]
        
        # Pivoter pour avoir une ligne par hash avec les classifications de chaque modèle
        pivot_results = filtered_results.pivot(index='hash', columns='model', values='classification')
        
        # Appliquer la logique de classification
        def get_final_classification(row):
            if row['phi'] == 'NONOK' and row['deepseek'] == 'NONOK':
                return 'NONOK'
            elif row['phi3'] == 'OK' and row['mistral'] == 'OK':
                return 'OK'
            else:
                # Compter les occurrences de chaque classification
                counts = row.value_counts()
                # Retourner la classification majoritaire
                return counts.index[0]
        
        pivot_results['final_classification'] = pivot_results.apply(get_final_classification, axis=1)
        
        # Récupérer les données de base avec poi_present pour le premier modèle
        base_data = pd.read_sql_query(f"SELECT hash, poi_present FROM results WHERE model = '{models[0]}'", conn)
        
        # Joindre avec pivot_results
        final_results = pivot_results.merge(base_data, on='hash', how='left')

        # Ajouter la colonne correct_prediction
        final_results['correct_prediction'] = (
            (final_results['poi_present'] == 1) & (final_results['final_classification'] == 'NONOK') |
            (final_results['poi_present'] == 0) & (final_results['final_classification'] == 'OK')
        )

        # Calculer les statistiques globales
        total_samples = len(final_results)
        correct_predictions = final_results['correct_prediction'].sum()
        accuracy = (correct_predictions / total_samples) * 100
        
        classification_dist = final_results['final_classification'].value_counts()
        
        print("\nRésultats de l'analyse combinée:")
        print(f"Nombre total d'échantillons: {total_samples}")
        print(f"Nombre de prédictions correctes: {correct_predictions}")
        print(f"Précision globale: {accuracy:.2f}%")
        
        print("\nDistribution des classifications:")
        for class_name, count in classification_dist.items():
            percentage = (count / total_samples) * 100
            print(f"{class_name}: {count} ({percentage:.2f}%)")
        
        print("\nAnalyse détaillée:")
        for classification in ['NONOK', 'OK', 'UNKNOWN']:
            subset = final_results[final_results['final_classification'] == classification]
            if len(subset) > 0:
                correct = subset['correct_prediction'].sum()
                accuracy = (correct / len(subset)) * 100
                print(f"\nPour la classification {classification}:")
                print(f"Nombre total: {len(subset)}")
                print(f"Prédictions correctes: {correct}")
                print(f"Précision: {accuracy:.2f}%")
        
        print("\nAnalyse détaillée des POI:")
        
        poi_emails = final_results[final_results['poi_present'] == 1]
        if len(poi_emails) > 0:
            correct_poi = poi_emails[poi_emails['final_classification'] == 'NONOK']
            print(f"\nEmails contenant des POI (poi_present = 1):")
            print(f"Nombre total: {len(poi_emails)}")
            print(f"Correctement identifiés comme NONOK: {len(correct_poi)}")
            print(f"Précision: {(len(correct_poi) / len(poi_emails) * 100):.2f}%")
        
        non_poi_emails = final_results[final_results['poi_present'] == 0]
        if len(non_poi_emails) > 0:
            correct_non_poi = non_poi_emails[non_poi_emails['final_classification'] == 'OK']
            print(f"\nEmails sans POI (poi_present = 0):")
            print(f"Nombre total: {len(non_poi_emails)}")
            print(f"Correctement identifiés comme OK: {len(correct_non_poi)}")
            print(f"Précision: {(len(correct_non_poi) / len(non_poi_emails) * 100):.2f}%")
        
        
    except Exception as e:
        print(f"Une erreur est survenue lors de la récupération des données: {str(e)}")
    finally:
        conn.close()

def supprimer_doublons():
    try:
        conn = sqlite3.connect('email_classifications.db')
        cursor = conn.cursor()
        
        # Identifie et supprime les doublons en gardant la première occurrence
        cursor.execute("""
            DELETE FROM results 
            WHERE rowid NOT IN (
                SELECT MIN(rowid)
                FROM results
                GROUP BY hash, model
            )
        """)
        
        # Récupère le nombre de lignes supprimées
        deleted_rows = cursor.rowcount
        
        conn.commit()
        print(f"Nombre de doublons supprimés: {deleted_rows}")
        
    except Exception as e:
        print(f"Une erreur est survenue: {str(e)}")
    finally:
        conn.close()

def visualize_results():
    """
    Visualise les résultats des différents modèles avec Matplotlib
    """
    models = ['phi', 'phi3', 'mistral', 'deepseek']
    conn = sqlite3.connect('email_classifications.db')
    
    try:
        # Création d'une figure avec 4 sous-graphiques en barres
        plt.style.use('bmh')  # Style intégré de matplotlib plus moderne
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))

        # Pour chaque modèle
        for idx, (model, ax) in enumerate(zip(models, axes.flat)):
            results = pd.read_sql_query(f"SELECT * FROM results WHERE model = '{model}'", conn)
            
            # Calcul des métriques
            poi_emails = results[results['poi_present'] == 1]
            poi_accuracy = (poi_emails['classification'] == 'NONOK').sum() / len(poi_emails) * 100
            
            nonpoi_emails = results[results['poi_present'] == 0]
            nonpoi_accuracy = (nonpoi_emails['classification'] == 'OK').sum() / len(nonpoi_emails) * 100
            
            avg_time = results['response_time'].mean() * 7.5
            avg_length = results['response_length'].mean()
            
            # Création des barres
            x = np.arange(2)
            accuracies = [poi_accuracy, nonpoi_accuracy]
            bars = ax.bar(x, accuracies, color=['#e74c3c', '#2ecc71'], width=0.5)
            
            # Personnalisation du graphique
            ax.set_ylim(0, 100)
            ax.set_xticks(x)
            ax.set_xticklabels(['Emails avec POI\nbien identifiés', 'Emails sans POI\nbien identifiés'], 
                              fontsize=10)
            
            # Grille horizontale uniquement et fond blanc
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)  # Mettre la grille en arrière-plan
            ax.set_facecolor('white')  # Fond blanc
            
            # Ajout des valeurs sur les barres
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', 
                       ha='center', va='bottom',
                       fontsize=12, fontweight='bold')
            
            # Titre avec informations
            title = f'{model.upper()}\n'
            title += f'Temps moyen: {avg_time:.2f}s - Longueur moyenne: {avg_length:.0f} car.'
            ax.set_title(title, pad=20, fontsize=12, fontweight='bold')

            # Ajout de l'étiquette de l'axe y
            ax.set_ylabel('Pourcentage (%)', fontsize=10)

        # Ajout d'une légende commune plus compacte
        legend_elements = [
            plt.Rectangle((0,0), 1, 1, facecolor='#e74c3c', 
                         label='Emails avec POI correctement identifiés comme NONOK'),
            plt.Rectangle((0,0), 1, 1, facecolor='#2ecc71', 
                         label='Emails sans POI correctement identifiés comme OK')
        ]
        fig.legend(handles=legend_elements, 
                  loc='center', 
                  bbox_to_anchor=(0.5, 0.02),
                  ncol=2, 
                  fontsize=11,
                  frameon=True,
                  edgecolor='black',
                  bbox_transform=fig.transFigure)

        # Fond blanc pour toute la figure
        fig.patch.set_facecolor('white')

        # Ajustement de la mise en page
        plt.tight_layout()
        # Ajuster l'espace pour la légende
        plt.subplots_adjust(bottom=0.15)
        
        # Sauvegarde du graphique
        plt.savefig('model_analysis.png', bbox_inches='tight', dpi=300, facecolor='white')
        
        # Affichage du graphique
        plt.show()
        
    except Exception as e:
        print(f"Une erreur est survenue: {str(e)}")
    finally:
        conn.close()

def visualize_combined_results():
    """
    Visualise les résultats de l'analyse combinée des modèles
    """
    conn = sqlite3.connect('email_classifications.db')
    
    try:
        # Récupération des données de base
        models = ['phi', 'phi3', 'mistral', 'deepseek', 'phishing']
        combined_results = pd.DataFrame()
        
        for model in models:
            results = pd.read_sql_query(f"SELECT * FROM results WHERE model = '{model}'", conn)
            results['model'] = model
            combined_results = pd.concat([combined_results, results])
        
        # Filtrer pour ne garder que les hashs avec les 4 modèles
        hash_counts = combined_results.groupby('hash').size()
        valid_hashes = hash_counts[hash_counts == 4].index
        filtered_results = combined_results[combined_results['hash'].isin(valid_hashes)]
        
        # Pivoter pour avoir une ligne par hash avec les classifications de chaque modèle
        pivot_results = filtered_results.pivot(index='hash', columns='model', values='classification')
        
        # Appliquer la logique de classification combinée
        def get_final_classification(row):
            if row['phi'] == 'NONOK' or row['deepseek'] == 'NONOK':
                return 'NONOK'
            elif row['phi3'] == 'OK' or row['mistral'] == 'OK':
                return 'OK'
            else:
                return 'UNKNOWN'
        
        pivot_results['final_classification'] = pivot_results.apply(get_final_classification, axis=1)
        
        # Récupérer les données de base avec poi_present
        base_data = pd.read_sql_query(f"SELECT hash, poi_present FROM results WHERE model = '{models[0]}'", conn)
        
        # Joindre avec pivot_results
        final_results = pivot_results.merge(base_data, on='hash', how='left')
        
        # Création de la figure
        plt.figure(figsize=(12, 10))
        
        # Calcul des métriques pour les emails avec POI
        poi_emails = final_results[final_results['poi_present'] == 1]
        poi_accuracy = (poi_emails['final_classification'] == 'NONOK').sum() / len(poi_emails) * 100
        
        # Calcul des métriques pour les emails sans POI
        nonpoi_emails = final_results[final_results['poi_present'] == 0]
        nonpoi_accuracy = (nonpoi_emails['final_classification'] == 'OK').sum() / len(nonpoi_emails) * 100
        
        # Calcul de la précision globale
        correct_predictions = (
            ((final_results['poi_present'] == 1) & (final_results['final_classification'] == 'NONOK')) |
            ((final_results['poi_present'] == 0) & (final_results['final_classification'] == 'OK'))
        ).sum()
        global_accuracy = (correct_predictions / len(final_results)) * 100
        
        # Création des barres
        x = np.arange(3)
        accuracies = [poi_accuracy, nonpoi_accuracy, global_accuracy]
        colors = ['#e74c3c', '#2ecc71', '#3498db']
        labels = ['Emails avec POI\nbien identifiés', 'Emails sans POI\nbien identifiés', 'Précision globale']
        
        bars = plt.bar(x, accuracies, color=colors, width=0.6)
        
        # Personnalisation du graphique
        plt.ylim(0, 100)
        plt.xticks(x, labels, fontsize=12)
        plt.ylabel('Pourcentage (%)', fontsize=12)
        plt.grid(True, axis='y', alpha=0.3, linestyle='--')
        plt.gca().set_axisbelow(True)
        plt.gca().set_facecolor('white')
        
        # Ajout des valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', 
                   ha='center', va='bottom',
                   fontsize=14, fontweight='bold')
        
        # Titre
        plt.title('Analyse des résultats combinés\n(Stratégie: phi/deepseek pour NONOK, phi3/mistral pour OK)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # # Distribution des classifications
        # classification_counts = final_results['final_classification'].value_counts()
        # classification_text = "Distribution des classifications:\n"
        # for cls, count in classification_counts.items():
        #     percentage = (count / len(final_results)) * 100
        #     classification_text += f"{cls}: {count} ({percentage:.1f}%)\n"
        
        # Ajout d'un texte explicatif
        # plt.figtext(0.15, 0.01, classification_text, fontsize=12, 
        #            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Ajout d'un texte explicatif sur la stratégie
        strategy_text = "Stratégie de classification combinée:\n"
        strategy_text += "- Si phi OU deepseek détectent NONOK → NONOK\n"
        strategy_text += "- Sinon, si phi3 OU mistral détectent OK → OK\n"
        strategy_text += "- Sinon → UNKNOWN"
        
        plt.figtext(0.41, 0.10, strategy_text, fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Ajustement de la mise en page
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        
        # Sauvegarde du graphique
        plt.savefig('combined_analysis.png', bbox_inches='tight', dpi=300, facecolor='white')
        
        # Affichage du graphique
        plt.show()
        
    except Exception as e:
        print(f"Une erreur est survenue: {str(e)}")
    finally:
        conn.close()

def visualize_model_comparisons():
    """
    Crée des visualisations comparatives entre les différents modèles pour le rapport
    """
    models = ['phi', 'phi3', 'mistral', 'deepseek', 'phishing', 'bert']
    conn = sqlite3.connect('email_classifications.db')
    
    try:
        
        # Création d'une grille 2x3 pour les camemberts
        fig1, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig1.suptitle('Distribution des classifications (OK vs NONOK)', fontsize=18, fontweight='bold', y=0.98)
        
        # Stockage des données pour le graphique en barres
        accuracies = []
        poi_accuracies = []
        nonpoi_accuracies = []
        
        # Pour chaque modèle - Camemberts
        for idx, model in enumerate(models):
            # Calcul de la position dans la grille 2x3
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            results = pd.read_sql_query(f"SELECT * FROM results WHERE model = '{model}'", conn)
            
            # Calcul des métriques
            total = len(results)
            print(f"Total pour {model}: {total}")
            ok_count = len(results[results['classification'] == 'OK'])
            nonok_count = len(results[results['classification'] == 'NONOK'])
            
            # Calcul de la précision pour le graphique en barres
            correct_predictions = (
                ((results['poi_present'] == 1) & (results['classification'] == 'NONOK')) |
                ((results['poi_present'] == 0) & (results['classification'] == 'OK'))
            ).sum()
            accuracy = (correct_predictions / total) * 100
            accuracies.append(accuracy)
            
            # Calcul des précisions POI/non-POI
            poi_emails = results[results['poi_present'] == 1]
            poi_accuracy = (poi_emails['classification'] == 'NONOK').sum() / len(poi_emails) * 100
            poi_accuracies.append(poi_accuracy)
            
            nonpoi_emails = results[results['poi_present'] == 0]
            nonpoi_accuracy = (nonpoi_emails['classification'] == 'OK').sum() / len(nonpoi_emails) * 100
            nonpoi_accuracies.append(nonpoi_accuracy)
            
            # Création du camembert
            sizes = [ok_count, nonok_count]
            labels = ['OK', 'NONOK']
            colors = ['#2ecc71', '#e74c3c']
            explode = (0.05, 0.05)  # Légère séparation des segments
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                            startangle=90, shadow=True, explode=explode)
            
            # Titre avec le nom du modèle
            ax.set_title(f'{model.upper()}', fontsize=14, fontweight='bold', pad=10)
            
            # Style des labels
            plt.setp(autotexts, size=10, weight="bold")
            plt.setp(texts, size=12)
        
        # Cacher le dernier graphique vide si nécessaire
        if len(models) < 6:
            axes[1, -1].set_visible(False)
        
        # Ajustement de la mise en page pour les camemberts
        fig1.tight_layout()
        fig1.savefig('model_distributions.png', bbox_inches='tight', dpi=300, facecolor='white')
        
        # 2. Graphique en barres comparant les précisions
        fig2, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(models))
        width = 0.25
        
        # Barres pour la précision globale
        bars1 = ax.bar(x - width, accuracies, width, label='Précision globale', color='#3498db')
        # Barres pour la précision POI
        bars2 = ax.bar(x, poi_accuracies, width, label='Détection des POI', color='#e74c3c')
        # Barres pour la précision non-POI
        bars3 = ax.bar(x + width, nonpoi_accuracies, width, label='Détection des non-POI', color='#2ecc71')
        
        # Personnalisation du graphique
        ax.set_title('Comparaison des précisions entre modèles', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models], fontsize=12)
        ax.set_ylabel('Précision (%)', fontsize=12)
        ax.set_ylim(0, 100)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.legend(fontsize=12)
        
        # Ajout des valeurs sur les barres
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom',
                       fontsize=10, fontweight='bold')
        
        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)
        
        # Ajustement de la mise en page pour le graphique en barres
        fig2.tight_layout()
        fig2.savefig('model_accuracy_comparison.png', bbox_inches='tight', dpi=300, facecolor='white')
        
        # Affichage des graphiques
        plt.show()
        
    except Exception as e:
        print(f"Une erreur est survenue: {str(e)}")
    finally:
        conn.close()

def combine_phi_and_phishing():
    """
    Combine les résultats des modèles phi et phishing selon les règles suivantes:
    - Si les deux disent NONOK -> NONOK 
    - Si les deux disent OK -> OK
    - Si Phi dit NONOK -> NONOK
    - Si phishing dit OK -> OK
    - Si Phi dit NONOK et phishing dit OK -> NONOK
    """
    conn = sqlite3.connect('email_classifications.db')
    try:
        # Récupérer les résultats des deux modèles
        phi_results = pd.read_sql_query("SELECT * FROM results WHERE model = 'phi'", conn)
        phishing_results = pd.read_sql_query("SELECT * FROM results WHERE model = 'phishing'", conn)
        
        # Fusionner les résultats sur le hash
        combined = pd.merge(phi_results, phishing_results, 
                          on=['hash', 'sender', 'subject', 'body', 'poi_present'],
                          suffixes=('_phi', '_phishing'))
        
        print(f"\nNombre total d'emails analysés: {len(combined)}")
        
        # Appliquer les règles de classification
        def get_final_classification(row):
            phi_class = row['classification_phi']
            phishing_class = row['classification_phishing']
            
            if phi_class == 'NONOK' and phishing_class == 'NONOK':
                return 'NONOK'
            elif phi_class == 'OK' and phishing_class == 'OK':
                return 'OK'
            elif phi_class == 'NONOK':
                return 'NONOK'
            elif phishing_class == 'OK':
                return 'OK'
            else:
                return 'NONOK'  # Par défaut NONOK si Phi dit NONOK et phishing dit OK
        
        combined['final_classification'] = combined.apply(get_final_classification, axis=1)
        
        # Calculer les métriques
        total = len(combined)
        correct_predictions = (
            ((combined['poi_present'] == 1) & (combined['final_classification'] == 'NONOK')) |
            ((combined['poi_present'] == 0) & (combined['final_classification'] == 'OK'))
        ).sum()
        
        accuracy = (correct_predictions / total) * 100
        
        # Calculer les précisions pour les POI et non-POI
        poi_emails = combined[combined['poi_present'] == 1]
        poi_accuracy = (poi_emails['final_classification'] == 'NONOK').sum() / len(poi_emails) * 100
        
        nonpoi_emails = combined[combined['poi_present'] == 0]
        nonpoi_accuracy = (nonpoi_emails['final_classification'] == 'OK').sum() / len(nonpoi_emails) * 100
        
        # Créer une figure avec 2 sous-graphiques
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Analyse combinée PHI + PHISHING', fontsize=16, fontweight='bold')
        
        # 1. Graphique des précisions
        metrics = ['Globale', 'POI', 'Non-POI']
        values = [accuracy, poi_accuracy, nonpoi_accuracy]
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        bars = ax1.bar(metrics, values, color=colors)
        ax1.set_title('Précisions', fontsize=14, pad=10)
        ax1.set_ylim(0, 100)
        ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylabel('Pourcentage (%)')
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
            
        # 2. Camembert de distribution des classifications
        class_dist = combined['final_classification'].value_counts()
        
        # Créer un explode dynamique basé sur le nombre de catégories
        explode = [0.05] * len(class_dist)
        
        # Déterminer les couleurs en fonction des catégories présentes
        colors = []
        for category in class_dist.index:
            if category == 'OK':
                colors.append('#2ecc71')  # vert pour OK
            else:
                colors.append('#e74c3c')  # rouge pour NONOK
        
        ax2.pie(class_dist.values, labels=class_dist.index, autopct='%1.1f%%',
                colors=colors, explode=explode,
                shadow=True, startangle=90)
        ax2.set_title('Distribution des classifications', fontsize=14, pad=10)
        
        plt.tight_layout()
        plt.savefig('combined_phi_phishing.png', bbox_inches='tight', dpi=300, facecolor='white')
        plt.show()
        
        # Afficher les résultats textuels
        print("\nRésultats de la combinaison PHI + PHISHING:")
        print(f"Précision globale: {accuracy:.1f}%")
        print(f"Détection des POI: {poi_accuracy:.1f}%")
        print(f"Détection des non-POI: {nonpoi_accuracy:.1f}%")
        
        print("\nDistribution des classifications:")
        for cls, count in class_dist.items():
            print(f"{cls}: {count} ({count/total*100:.1f}%)")
        
        # Matrice de confusion entre les deux modèles
        print("\nMatrice de confusion entre PHI et PHISHING:")
        confusion = pd.crosstab(combined['classification_phi'], 
                              combined['classification_phishing'],
                              margins=True)
        print(confusion)
        
        return combined
        
    except Exception as e:
        print(f"Une erreur est survenue: {str(e)}")
    finally:
        conn.close()

def optimize_model_combination():
    """
    Combine les résultats des modèles de manière optimisée avec une stratégie équilibrée:
    
    - Si DEEPSEEK et PHI disent tous les deux NONOK -> NONOK (haute confiance POI)
    - Si MISTRAL et PHISHING disent tous les deux OK -> OK (haute confiance non-POI)
    - Si 3 modèles ou plus sont d'accord sur une classification -> suivre la majorité
    - Sinon, pour les cas ambigus:
      - Utiliser DEEPSEEK (96% POI) pour détecter POI
      - Utiliser MISTRAL (79% non-POI) pour détecter non-POI
    """
    conn = sqlite3.connect('email_classifications.db')
    try:
        # Récupérer les résultats des modèles
        deepseek = pd.read_sql_query("SELECT hash, classification, poi_present FROM results WHERE model = 'deepseek'", conn)
        phi = pd.read_sql_query("SELECT hash, classification FROM results WHERE model = 'phi'", conn)
        mistral = pd.read_sql_query("SELECT hash, classification FROM results WHERE model = 'mistral'", conn)
        phishing = pd.read_sql_query("SELECT hash, classification FROM results WHERE model = 'phishing'", conn)
        
        # Renommer les colonnes avant la fusion pour éviter les conflits
        deepseek = deepseek.rename(columns={'classification': 'classification_deepseek'})
        phi = phi.rename(columns={'classification': 'classification_phi'})
        mistral = mistral.rename(columns={'classification': 'classification_mistral'})
        phishing = phishing.rename(columns={'classification': 'classification_phishing'})
        
        # Fusionner les résultats sur le hash
        combined = deepseek.merge(phi, on='hash', how='left')
        combined = combined.merge(mistral, on='hash', how='left')
        combined = combined.merge(phishing, on='hash', how='left')
        
        print(f"\nNombre total d'emails analysés: {len(combined)}")
        
        # Appliquer la stratégie de classification équilibrée
        def get_optimized_classification(row):
            # Compter les classifications
            classifications = [
                row['classification_deepseek'], 
                row['classification_phi'],
                row['classification_mistral'], 
                row['classification_phishing']
            ]
            
            ok_count = classifications.count('OK')
            nonok_count = classifications.count('NONOK')
            
            # Règle 1: Haute confiance pour POI - DEEPSEEK et PHI sont d'accord
            if (row['classification_deepseek'] == 'NONOK' and 
                row['classification_phi'] == 'NONOK'):
                return 'NONOK'
                
            # Règle 2: Haute confiance pour non-POI - MISTRAL et PHISHING sont d'accord
            elif (row['classification_mistral'] == 'OK' and 
                  row['classification_phishing'] == 'OK'):
                return 'OK'
                
            # Règle 3: Suivre la majorité si 3 modèles ou plus sont d'accord
            elif ok_count >= 3:
                return 'OK'
            elif nonok_count >= 3:
                return 'NONOK'
                
            # Règle 4: Pour les cas ambigus (2-2), utiliser les modèles les plus précis
            elif row['classification_deepseek'] == 'NONOK':
                # DEEPSEEK est meilleur pour POI (96%)
                return 'NONOK'
            elif row['classification_mistral'] == 'OK':
                # MISTRAL est meilleur pour non-POI (79%)
                return 'OK'
            else:
                # Dernier recours - vérifier le modèle PHISHING
                return row['classification_phishing']
        
        combined['final_classification'] = combined.apply(get_optimized_classification, axis=1)
        
        # Calculer les métriques
        total = len(combined)
        correct_predictions = (
            ((combined['poi_present'] == 1) & (combined['final_classification'] == 'NONOK')) |
            ((combined['poi_present'] == 0) & (combined['final_classification'] == 'OK'))
        ).sum()
        
        accuracy = (correct_predictions / total) * 100
        
        # Calculer les précisions pour les POI et non-POI
        poi_emails = combined[combined['poi_present'] == 1]
        poi_accuracy = (poi_emails['final_classification'] == 'NONOK').sum() / len(poi_emails) * 100
        
        nonpoi_emails = combined[combined['poi_present'] == 0]
        nonpoi_accuracy = (nonpoi_emails['final_classification'] == 'OK').sum() / len(nonpoi_emails) * 100
        
        # Créer une figure avec 2 sous-graphiques
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Analyse de la combinaison optimisée et équilibrée', fontsize=16, fontweight='bold')
        
        # 1. Graphique des précisions
        metrics = ['Globale', 'POI', 'Non-POI']
        values = [accuracy, poi_accuracy, nonpoi_accuracy]
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        bars = ax1.bar(metrics, values, color=colors)
        ax1.set_title('Précisions', fontsize=14, pad=10)
        ax1.set_ylim(0, 100)
        ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylabel('Pourcentage (%)')
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
            
        # 2. Camembert de distribution des classifications
        class_dist = combined['final_classification'].value_counts()
        
        # Créer un explode dynamique basé sur le nombre de catégories
        explode = [0.05] * len(class_dist)
        
        # Déterminer les couleurs en fonction des catégories présentes
        colors = []
        for category in class_dist.index:
            if category == 'OK':
                colors.append('#2ecc71')  # vert pour OK
            else:
                colors.append('#e74c3c')  # rouge pour NONOK
        
        ax2.pie(class_dist.values, labels=class_dist.index, autopct='%1.1f%%',
                colors=colors, explode=explode,
                shadow=True, startangle=90)
        ax2.set_title('Distribution des classifications', fontsize=14, pad=10)
        
        # Ajouter le texte des règles pour expliquer la stratégie
        strategy_text = "Stratégie de classification équilibrée:\n"
        strategy_text += "1. Si DEEPSEEK et PHI disent NONOK → NONOK (haute confiance POI)\n"
        strategy_text += "2. Si MISTRAL et PHISHING disent OK → OK (haute confiance non-POI)\n"
        strategy_text += "3. Si 3+ modèles sont d'accord → Suivre la majorité\n"
        strategy_text += "4. Cas ambigus: Utiliser le modèle le plus performant par catégorie"
        
        plt.figtext(0.5, 0.01, strategy_text, ha="center", fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.5"))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.22)
        plt.savefig('optimized_balanced_combination.png', bbox_inches='tight', dpi=300, facecolor='white')
        plt.show()
        
        # Afficher les résultats textuels
        print("\nRésultats de la combinaison optimisée et équilibrée:")
        print(f"Précision globale: {accuracy:.1f}%")
        print(f"Détection des POI: {poi_accuracy:.1f}%")
        print(f"Détection des non-POI: {nonpoi_accuracy:.1f}%")
        
        print("\nDistribution des classifications:")
        for cls, count in class_dist.items():
            print(f"{cls}: {count} ({count/total*100:.1f}%)")
        
        # Matrice de confusion entre les prédictions et la réalité
        print("\nMatrice de confusion entre prédictions et réalité:")
        confusion = pd.crosstab(
            combined['poi_present'].map({1: 'POI présent', 0: 'POI absent'}),
            combined['final_classification'],
            margins=True
        )
        print(confusion)
        
        return combined
        
    except Exception as e:
        print(f"Une erreur est survenue: {str(e)}")
        raise e  # Relever l'erreur pour voir la stack trace complète
    finally:
        conn.close()

if __name__ == "__main__":
    # remove_ok_classifications()
    # analyze_accuracy()
    # analyze_accuracy_combined()
    # analyze_accuracy_combined_2()
    # supprimer_doublons()
    # visualize_results()
    # visualize_combined_results()
    visualize_model_comparisons()   
    # combine_phi_and_phishing()
    # optimize_model_combination()