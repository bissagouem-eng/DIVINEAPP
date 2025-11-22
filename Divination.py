# QUANTUM QUINTE AI - STREAMLIT CLOUD COMPATIBLE
import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
import PyPDF2
from datetime import datetime

# ========== CORE PMUB KNOWLEDGE BASE ==========
class PMUBExpertSystem:
    def __init__(self):
        self.game_definitions = self._build_game_definitions()
        self.french_patterns = self._build_french_patterns()
        self.horse_racing_terms = self._build_racing_terms()
        
    def _build_game_definitions(self):
        """Comprehensive PMUB game definitions"""
        return {
            'TIERCE': {
                'description': 'Trouver les 3 premiers chevaux dans l\'ordre',
                'horses_required': 3,
                'bet_types': ['Ordre', 'Désordre', 'Bonus 4'],
                'typical_field': '8-16 chevaux',
                'key_indicators': ['TIERCÉ', '3 NUMÉROS', 'ORDRE/DÉSORDRE', 'TROIS PREMIERS'],
                'combination_rules': '3 chevaux exacts dans l\'ordre'
            },
            'QUARTE': {
                'description': 'Trouver les 4 premiers chevaux dans l\'ordre',
                'horses_required': 4,
                'bet_types': ['Ordre', 'Désordre'],
                'typical_field': '10-16 chevaux',
                'key_indicators': ['QUARTÉ', '4 NUMÉROS', 'QUATRE PREMIERS'],
                'combination_rules': '4 chevaux exacts dans l\'ordre'
            },
            'QUARTE_PLUS': {
                'description': 'Quarté +1 - Les 4 premiers + 1 cheval supplémentaire',
                'horses_required': 5,
                'bet_types': ['Ordre', 'Désordre avec base'],
                'typical_field': '12-18 chevaux',
                'key_indicators': ['QUARTÉ+1', 'QUARTÉ +1', 'QUARTE+1', '4+1 NUMÉROS'],
                'combination_rules': '4 chevaux dans l\'ordre + 1 cheval base'
            },
            'QUINTE': {
                'description': 'Trouver les 5 premiers chevaux dans l\'ordre',
                'horses_required': 5,
                'bet_types': ['Ordre', 'Désordre'],
                'typical_field': '14-20 chevaux',
                'key_indicators': ['QUINTÉ', '5 NUMÉROS', 'CINQ PREMIERS'],
                'combination_rules': '5 chevaux exacts dans l\'ordre'
            },
            'QUINTE_PLUS': {
                'description': 'Quinté +1 - Les 5 premiers + 1 cheval supplémentaire',
                'horses_required': 6,
                'bet_types': ['Ordre', 'Désordre avec base'],
                'typical_field': '16-24 chevaux',
                'key_indicators': ['QUINTÉ+1', 'QUINTÉ +1', 'QUINTE+1', '5+1 NUMÉROS'],
                'combination_rules': '5 chevaux dans l\'ordre + 1 cheval base'
            }
        }
    
    def _build_french_patterns(self):
        """French language patterns specific to PMUB"""
        return {
            'game_titles': [
                (r'TIERCÉ', 'TIERCE'),
                (r'TIERCE', 'TIERCE'),
                (r'QUARTÉ', 'QUARTE'),
                (r'QUARTE', 'QUARTE'),
                (r'QUARTÉ\s*\+\s*1', 'QUARTE_PLUS'),
                (r'QUARTE\s*\+\s*1', 'QUARTE_PLUS'),
                (r'QUINTÉ', 'QUINTE'),
                (r'QUINTE', 'QUINTE'),
                (r'QUINTÉ\s*\+\s*1', 'QUINTE_PLUS'),
                (r'QUINTE\s*\+\s*1', 'QUINTE_PLUS')
            ],
            'bet_types': [
                (r'ORDRE', 'Ordre'),
                (r'DÉSORDRE', 'Désordre'),
                (r'DESORDRE', 'Désordre'),
                (r'BONUS', 'Bonus'),
                (r'COUPLÉ', 'Couplé'),
                (r'COUPLE', 'Couplé'),
                (r'MULTI', 'Multi')
            ],
            'race_terms': [
                (r'COURSE', 'Course'),
                (r'PRIX', 'Prix'),
                (r'HIPPODROME', 'Hippodrome'),
                (r'CHEVAUX', 'Chevaux'),
                (r'PARTANTS', 'Partants'),
                (r'RÉSULTAT', 'Résultat'),
                (r'RESULTAT', 'Résultat'),
                (r'ARRIVÉE', 'Arrivée'),
                (r'ARRIVEE', 'Arrivée')
            ]
        }
    
    def _build_racing_terms(self):
        """French horse racing terminology"""
        return {
            'positions': ['PREMIER', 'DEUXIÈME', 'TROISIÈME', 'QUATRIÈME', 'CINQUIÈME', 'SIXIÈME'],
            'conditions': ['OFFICIEL', 'OFFICIELLE', 'HANDICAP', 'PLAT', 'OBSTACLE', 'HAIES', 'STEEPLECHASE'],
            'locations': ['VINCENNES', 'LONGCHAMP', 'CHANTILLY', 'DEAUVILLE', 'MAISONS-LAFFITTE']
        }

# ========== ENHANCED PDF ANALYZER ==========
class PMUBAnalyzer:
    def __init__(self):
        self.expert = PMUBExpertSystem()
        self.detection_weights = self._build_detection_weights()
        
    def _build_detection_weights(self):
        """Weighted detection system for accurate game identification"""
        return {
            'direct_title': 10,
            'game_description': 8,
            'horse_count_context': 7,
            'bet_type_mention': 6,
            'result_structure': 5,
            'field_size': 4
        }
    
    def analyze_pdf(self, pdf_file):
        """Comprehensive PDF analysis using PyPDF2"""
        try:
            analysis = self._initialize_analysis(pdf_file)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            full_text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text() or ""
                full_text += page_text + "\n"
                
                # Analyze page content
                self._analyze_page_content(page_text, page_num, analysis)
            
            # Final analysis
            analysis['full_text'] = full_text
            analysis['detected_game'] = self._determine_game_type(analysis)
            analysis['confidence'] = self._calculate_confidence(analysis)
            analysis['analysis_timestamp'] = datetime.now().isoformat()
            
            return analysis
            
        except Exception as e:
            st.error(f"❌ PDF analysis error: {str(e)[:100]}...")
            return self._get_empty_analysis()
    
    def _initialize_analysis(self, pdf_file):
        """Initialize comprehensive analysis structure"""
        return {
            'filename': getattr(pdf_file, 'name', 'unknown'),
            'game_evidence': {game: 0 for game in self.expert.game_definitions},
            'detected_horses': [],
            'race_info': {},
            'bet_types_found': [],
            'pages_analyzed': 0,
            'total_horses_detected': 0,
            'detection_details': [],
            'text_samples': []
        }
    
    def _analyze_page_content(self, text, page_num, analysis):
        """Analyze page content for PMUB-specific patterns"""
        if not text or len(text.strip()) < 10:
            return
            
        analysis['pages_analyzed'] += 1
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines):
            clean_line = line.strip()
            if len(clean_line) > 3:
                self._detect_game_evidence(clean_line, analysis, page_num, line_num)
                self._extract_horse_data(clean_line, analysis)
                self._extract_race_info(clean_line, analysis)
    
    def _detect_game_evidence(self, text, analysis, page_num, line_num):
        """Detect game type evidence with weighted scoring"""
        text_upper = text.upper()
        
        # Direct game title detection
        for pattern, game_type in self.expert.french_patterns['game_titles']:
            if re.search(pattern, text_upper, re.IGNORECASE):
                weight = self.detection_weights['direct_title']
                analysis['game_evidence'][game_type] += weight
                
                analysis['detection_details'].append({
                    'type': 'direct_title',
                    'game': game_type,
                    'text': text_upper[:80],
                    'weight': weight,
                    'page': page_num,
                    'line': line_num
                })
        
        # Game description detection
        for game_type, definition in self.expert.game_definitions.items():
            for indicator in definition['key_indicators']:
                if indicator in text_upper:
                    weight = self.detection_weights['game_description']
                    analysis['game_evidence'][game_type] += weight
                    
                    analysis['detection_details'].append({
                        'type': 'game_description',
                        'game': game_type,
                        'text': text_upper[:80],
                        'weight': weight,
                        'page': page_num,
                        'line': line_num
                    })
    
    def _extract_horse_data(self, text, analysis):
        """Extract horse information with multiple pattern matching"""
        # Enhanced horse pattern matching
        horse_patterns = [
            r'(\d{1,2})\s+([A-ZÀ-ÿ][a-zà-ÿ\s\'-]{2,})\s+([A-Z][a-zA-Z\s\-\.]+)',
            r'(\d{1,2})\s+([A-Z][A-Za-z\s\'-]{2,})',
            r'^(\d{1,2})\s+([A-Z][A-Za-z\s\']+)',
            r'\((\d{1,2})\)\s+([A-Z][A-Za-z\s\']+)'
        ]
        
        for pattern in horse_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    horse_number = int(match.group(1))
                    horse_name = match.group(2).strip()
                    
                    # Avoid duplicates
                    if not any(h['number'] == horse_number for h in analysis['detected_horses']):
                        horse_info = {
                            'number': horse_number,
                            'name': horse_name,
                            'details': match.group(3).strip() if len(match.groups()) > 2 else 'Non spécifié',
                            'source_text': text[:40] + "..." if len(text) > 40 else text,
                            'extraction_confidence': 'high' if len(match.groups()) > 2 else 'medium'
                        }
                        analysis['detected_horses'].append(horse_info)
                        analysis['total_horses_detected'] += 1
                except (ValueError, IndexError):
                    continue
    
    def _extract_race_info(self, text, analysis):
        """Extract race information"""
        text_upper = text.upper()
        
        # Race title pattern
        if 'PRIX' in text_upper and 'race_name' not in analysis['race_info']:
            analysis['race_info']['race_name'] = text.strip()[:100]
        
        # Date pattern (French format)
        date_pattern = r'(\d{1,2}/\d{1,2}/\d{4})'
        match = re.search(date_pattern, text)
        if match and 'date' not in analysis['race_info']:
            analysis['race_info']['date'] = match.group(1)
    
    def _determine_game_type(self, analysis):
        """Determine the most likely game type based on evidence"""
        evidence = analysis['game_evidence']
        
        # If we have strong direct evidence
        max_score = max(evidence.values())
        if max_score >= 5:
            for game_type, score in evidence.items():
                if score == max_score:
                    return game_type
        
        # Fallback based on horse count
        horse_count = analysis['total_horses_detected']
        
        if horse_count <= 10:
            return 'TIERCE'
        elif horse_count <= 14:
            return 'QUARTE'
        elif horse_count <= 16:
            return 'QUARTE_PLUS'
        elif horse_count <= 20:
            return 'QUINTE'
        else:
            return 'QUINTE_PLUS'
    
    def _calculate_confidence(self, analysis):
        """Calculate detection confidence score"""
        evidence = analysis['game_evidence']
        max_score = max(evidence.values()) if evidence else 0
        total_horses = analysis['total_horses_detected']
        
        if max_score >= 10:
            base_confidence = 0.9
        elif max_score >= 5:
            base_confidence = 0.7
        elif max_score >= 2:
            base_confidence = 0.5
        else:
            base_confidence = 0.3
        
        # Adjust based on horse data
        if total_horses >= 8:
            horse_boost = 0.2
        elif total_horses >= 4:
            horse_boost = 0.1
        else:
            horse_boost = 0
        
        return min(0.95, base_confidence + horse_boost)
    
    def _get_empty_analysis(self):
        """Return empty analysis structure"""
        return {
            'filename': 'unknown',
            'game_evidence': {game: 0 for game in self.expert.game_definitions},
            'detected_horses': [],
            'race_info': {},
            'bet_types_found': [],
            'pages_analyzed': 0,
            'total_horses_detected': 0,
            'detected_game': 'TIERCE',
            'confidence': 0.1,
            'analysis_timestamp': datetime.now().isoformat()
        }

# ========== COMBINATION GENERATOR ==========
class CombinationGenerator:
    def __init__(self):
        self.strategies = self._build_strategies()
    
    def _build_strategies(self):
        """Build combination generation strategies"""
        return {
            'TIERCE': {
                'name': 'Tierce Strategy',
                'description': '3 chevaux exacts dans l\'ordre',
                'base_combinations': 6
            },
            'QUARTE': {
                'name': 'Quarté Strategy', 
                'description': '4 chevaux dans l\'ordre',
                'base_combinations': 24
            },
            'QUARTE_PLUS': {
                'name': 'Quarté+1 Strategy',
                'description': '4 chevaux ordre + 1 base',
                'base_combinations': 120
            },
            'QUINTE': {
                'name': 'Quinté Strategy',
                'description': '5 chevaux dans l\'ordre',
                'base_combinations': 120
            },
            'QUINTE_PLUS': {
                'name': 'Quinté+1 Strategy',
                'description': '5 chevaux ordre + 1 base',
                'base_combinations': 720
            }
        }
    
    def generate_combinations(self, analysis):
        """Generate combinations based on detected game type"""
        game_type = analysis['detected_game']
        horses = analysis['detected_horses']
        
        if not horses:
            return {"error": "Aucun cheval détecté pour générer des combinaisons"}
        
        horse_numbers = [h['number'] for h in horses]
        strategy_info = self.strategies[game_type]
        
        # Generate sample combinations
        combinations = self._generate_sample_combinations(horse_numbers, game_type)
        
        return {
            'game_type': game_type,
            'strategy': strategy_info['name'],
            'total_horses': len(horses),
            'horse_numbers': sorted(horse_numbers),
            'sample_combinations': combinations,
            'strategy_description': strategy_info['description'],
            'recommendation': self._get_recommendation(game_type, len(horses))
        }
    
    def _generate_sample_combinations(self, numbers, game_type):
        """Generate sample combinations based on game type"""
        if len(numbers) < 3:
            return ["Pas assez de chevaux détectés"]
        
        if game_type == 'TIERCE':
            return [
                f"ORDRE: {numbers[0]} - {numbers[1]} - {numbers[2]}",
                f"ORDRE: {numbers[1]} - {numbers[0]} - {numbers[2]}", 
                f"DÉSORDRE: {numbers[0]}, {numbers[1]}, {numbers[2]}"
            ]
        elif game_type == 'QUARTE':
            return [
                f"ORDRE: {numbers[0]} - {numbers[1]} - {numbers[2]} - {numbers[3]}",
                f"DÉSORDRE: {numbers[0]}, {numbers[1]}, {numbers[2]}, {numbers[3]}"
            ]
        elif game_type == 'QUINTE':
            return [
                f"ORDRE: {numbers[0]} - {numbers[1]} - {numbers[2]} - {numbers[3]} - {numbers[4]}",
                f"DÉSORDRE: {numbers[0]}, {numbers[1]}, {numbers[2]}, {numbers[3]}, {numbers[4]}"
            ]
        else:
            return [f"Combinaisons pour {game_type} en cours de développement"]
    
    def _get_recommendation(self, game_type, horse_count):
        """Get strategy recommendation"""
        recommendations = {
            'TIERCE': f"Focus sur 3 chevaux forts parmi {horse_count} partants",
            'QUARTE': f"Sélectionnez 4 chevaux avec {horse_count} au total",
            'QUINTE': f"5 chevaux exacts dans l'ordre avec {horse_count} partants"
        }
        return recommendations.get(game_type, f"Stratégie {game_type} avec {horse_count} chevaux")

# ========== STREAMLIT APPLICATION ==========
def main():
    st.set_page_config(
        page_title="QUANTUM QUINTE AI - PMUB Expert",
        page_icon="🎯",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        background: linear-gradient(45deg, #FF6B00, #FF0000);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .pmub-card {
        background: rgba(255,107,0,0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #FF6B00;
    }
    .confidence-high { color: #00AA00; font-weight: bold; }
    .confidence-medium { color: #FFAA00; font-weight: bold; }
    .confidence-low { color: #FF0000; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-title">
    🎯 QUANTUM QUINTE AI
    </div>
    <div style="text-align: center; color: #666; margin-bottom: 2rem;">
    Système Expert PMUB : Tiercé • Quarté • Quinté • et leurs variantes
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = PMUBAnalyzer()
        st.session_state.current_analysis = None
        st.session_state.expert_system = PMUBExpertSystem()
        st.session_state.combination_generator = CombinationGenerator()
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📁 Upload PDF PMUB")
        uploaded_file = st.file_uploader(
            "Choisissez votre fichier PDF PMUB", 
            type=['pdf'],
            help="Upload un PDF de Tiercé, Quarté, Quarté+1, Quinté ou Quinté+1"
        )
        
        if uploaded_file:
            if uploaded_file != st.session_state.get('last_uploaded_file'):
                st.session_state.last_uploaded_file = uploaded_file
                
                with st.spinner("🔍 Analyse expert PMUB en cours..."):
                    analysis = st.session_state.analyzer.analyze_pdf(uploaded_file)
                    st.session_state.current_analysis = analysis
                
                display_expert_analysis(analysis)
    
    with col2:
        st.markdown("## 🏇 Guide PMUB")
        st.markdown("""
        <div class="pmub-card">
        **TIERCÉ**: 3 premiers dans l'ordre  
        **QUARTÉ**: 4 premiers chevaux  
        **QUARTÉ+1**: 4 premiers + 1 base  
        **QUINTÉ**: 5 premiers chevaux  
        **QUINTÉ+1**: 5 premiers + 1 base
        </div>
        """, unsafe_allow_html=True)
        
        # Game type selector for learning
        st.markdown("### 📚 Apprendre les jeux")
        selected_game = st.selectbox(
            "Sélectionnez un jeu:",
            list(st.session_state.expert_system.game_definitions.keys())
        )
        
        if selected_game:
            game_info = st.session_state.expert_system.game_definitions[selected_game]
            st.write(f"**Description:** {game_info['description']}")
            st.write(f"**Chevaux requis:** {game_info['horses_required']}")

def display_expert_analysis(analysis):
    """Display comprehensive analysis results"""
    st.success("✅ Analyse PMUB terminée avec succès!")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        confidence = analysis['confidence']
        if confidence > 0.7:
            icon, color = "🟢", "confidence-high"
        elif confidence > 0.4:
            icon, color = "🟡", "confidence-medium"
        else:
            icon, color = "🔴", "confidence-low"
        st.metric("Confiance", f"{icon} {confidence:.0%}")
    
    with col2:
        st.metric("Type de Jeu", analysis['detected_game'])
    
    with col3:
        st.metric("Chevaux Trouvés", analysis['total_horses_detected'])
    
    with col4:
        st.metric("Pages", analysis['pages_analyzed'])
    
    # Detailed analysis
    with st.expander("🔍 DÉTAILS DE L'ANALYSE", expanded=True):
        # Game evidence
        st.subheader("🎯 Preuves de Détection")
        evidence_data = []
        for game_type, score in analysis['game_evidence'].items():
            if score > 0:
                evidence_data.append({'Jeu': game_type, 'Score': score})
        
        if evidence_data:
            evidence_df = pd.DataFrame(evidence_data)
            st.dataframe(evidence_df.sort_values('Score', ascending=False), use_container_width=True)
        else:
            st.info("Aucune preuve directe trouvée - utilisation de l'analyse contextuelle")
        
        # Horse data
        if analysis['detected_horses']:
            st.subheader("🐎 Chevaux Détectés")
            horses_df = pd.DataFrame([{
                'Numéro': h['number'],
                'Nom': h['name'],
                'Détails': h['details'],
                'Confiance': h['extraction_confidence']
            } for h in analysis['detected_horses']])
            st.dataframe(horses_df.sort_values('Numéro'), use_container_width=True)
        else:
            st.warning("Aucun cheval détecté - vérifiez la qualité du PDF")
    
    # Generate combinations
    st.markdown("---")
    st.markdown("## 🎯 GÉNÉRER DES COMBINAISONS")
    
    if st.button("🚀 GÉNÉRER DES COMBINAISONS INTELLIGENTES", type="primary", use_container_width=True):
        generate_combinations_display(analysis)

def generate_combinations_display(analysis):
    """Generate and display PMUB combinations"""
    with st.spinner("🎰 Génération de combinaisons intelligentes..."):
        combinations = st.session_state.combination_generator.generate_combinations(analysis)
    
    if "error" in combinations:
        st.error(combinations["error"])
        return
    
    st.success(f"✅ Combinaisons {combinations['game_type']} générées!")
    
    # Display results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎯 Stratégie")
        st.write(f"**Type de Jeu:** {combinations['game_type']}")
        st.write(f"**Stratégie:** {combinations['strategy']}")
        st.write(f"**Description:** {combinations['strategy_description']}")
        st.write(f"**Recommandation:** {combinations['recommendation']}")
    
    with col2:
        st.markdown("### 📊 Chevaux Disponibles")
        st.write(f"Nombres: {', '.join(map(str, combinations['horse_numbers']))}")
        st.write(f"Total: {combinations['total_horses']} chevaux")
    
    # Display combinations
    st.markdown("### 🎰 Combinaisons Générées")
    for i, combo in enumerate(combinations['sample_combinations'], 1):
        st.write(f"{i}. {combo}")

if __name__ == "__main__":
    main()
