# QUANTUM QUINTE AI - COMPREHENSIVE PMUB ANALYZER
import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
import PyPDF2
import fitz  # PyMuPDF for better text extraction
import io
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
                'key_indicators': ['TIERCÉ', '3 NUMÉROS', 'ORDRE/DÉSORDRE', 'TROIS PREMIERS']
            },
            'QUARTE': {
                'description': 'Trouver les 4 premiers chevaux dans l\'ordre',
                'horses_required': 4,
                'bet_types': ['Ordre', 'Désordre'],
                'typical_field': '10-16 chevaux',
                'key_indicators': ['QUARTÉ', '4 NUMÉROS', 'QUATRE PREMIERS']
            },
            'QUARTE_PLUS': {
                'description': 'Quarté +1 - Les 4 premiers + 1 cheval supplémentaire',
                'horses_required': 5,
                'bet_types': ['Ordre', 'Désordre avec base'],
                'typical_field': '12-18 chevaux',
                'key_indicators': ['QUARTÉ+1', 'QUARTÉ +1', 'QUARTE+1', '4+1 NUMÉROS']
            },
            'QUINTE': {
                'description': 'Trouver les 5 premiers chevaux dans l\'ordre',
                'horses_required': 5,
                'bet_types': ['Ordre', 'Désordre'],
                'typical_field': '14-20 chevaux',
                'key_indicators': ['QUINTÉ', '5 NUMÉROS', 'CINQ PREMIERS']
            },
            'QUINTE_PLUS': {
                'description': 'Quinté +1 - Les 5 premiers + 1 cheval supplémentaire',
                'horses_required': 6,
                'bet_types': ['Ordre', 'Désordre avec base'],
                'typical_field': '16-24 chevaux',
                'key_indicators': ['QUINTÉ+1', 'QUINTÉ +1', 'QUINTE+1', '5+1 NUMÉROS']
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

# ========== ADVANCED PDF ANALYZER ==========
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
        """Comprehensive PDF analysis using both PyPDF2 and PyMuPDF"""
        try:
            # Try PyMuPDF first for better text extraction
            analysis = self._initialize_analysis(pdf_file)
            
            # Use PyMuPDF for superior text extraction
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            full_text = ""
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                page_text = page.get_text()
                full_text += page_text + "\n"
                
                # Analyze page content
                self._analyze_page_content(page_text, page_num, analysis)
            
            pdf_document.close()
            
            # Final analysis
            analysis['full_text'] = full_text
            analysis['detected_game'] = self._determine_game_type(analysis)
            analysis['confidence'] = self._calculate_confidence(analysis)
            analysis['analysis_timestamp'] = datetime.now().isoformat()
            
            return analysis
            
        except Exception as e:
            st.error(f"❌ Advanced PDF analysis failed: {e}")
            # Fallback to PyPDF2
            return self._fallback_analysis(pdf_file)
    
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
            'text_quality_score': 0,
            'detection_details': [],
            'raw_matches': []
        }
    
    def _analyze_page_content(self, text, page_num, analysis):
        """Analyze page content for PMUB-specific patterns"""
        analysis['pages_analyzed'] += 1
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines):
            clean_line = line.strip()
            if len(clean_line) > 5:  # Only analyze substantial lines
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
                    'text': text_upper[:100],
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
                        'text': text_upper[:100],
                        'weight': weight,
                        'page': page_num,
                        'line': line_num
                    })
        
        # Bet type detection
        for pattern, bet_type in self.expert.french_patterns['bet_types']:
            if re.search(pattern, text_upper):
                analysis['bet_types_found'].append(bet_type)
                analysis['game_evidence']['TIERCE'] += 2  # Most common for bet types
    
    def _extract_horse_data(self, text, analysis):
        """Extract horse information with French name patterns"""
        # French horse name pattern: Number + French Name (accented) + Trainer/Jockey
        horse_patterns = [
            r'(\d{1,2})\s+([A-ZÀ-ÿ][a-zà-ÿ\s\'-]+)\s+([A-Z][a-zA-Z\s\-\.]+)',
            r'(\d{1,2})\s+([A-ZÀ-ÿ][a-zà-ÿ\s\'-]+)',
            r'(\d{1,2})\s+([A-Z][A-Z\s\'-]+)\s+([A-Z]\.[A-Z]+)',
            r'^(\d{1,2})\s+([A-Z][a-zA-Z\s\'-]+)'
        ]
        
        for pattern in horse_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                horse_info = {
                    'number': int(match.group(1)),
                    'name': match.group(2).strip(),
                    'details': match.group(3).strip() if len(match.groups()) > 2 else 'Non spécifié',
                    'source_text': text[:50] + "..." if len(text) > 50 else text,
                    'extraction_confidence': 'high' if len(match.groups()) > 2 else 'medium'
                }
                
                # Avoid duplicates
                if not any(h['number'] == horse_info['number'] for h in analysis['detected_horses']):
                    analysis['detected_horses'].append(horse_info)
                    analysis['total_horses_detected'] += 1
    
    def _extract_race_info(self, text, analysis):
        """Extract race information"""
        # Race title pattern
        race_pattern = r'(PRIX|COURSE)\s+([A-ZÀ-ÿ][A-ZÀ-ÿ\s\'-]+)'
        match = re.search(race_pattern, text, re.IGNORECASE)
        if match and 'race_name' not in analysis['race_info']:
            analysis['race_info']['race_name'] = text.strip()
        
        # Date pattern
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
        
        # Fallback based on horse count and context
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
        
        # Base confidence on evidence strength
        if max_score >= 10:
            base_confidence = 0.9
        elif max_score >= 5:
            base_confidence = 0.7
        elif max_score >= 2:
            base_confidence = 0.5
        else:
            base_confidence = 0.3
        
        # Adjust based on horse data quality
        if total_horses >= 8:
            horse_boost = 0.2
        elif total_horses >= 4:
            horse_boost = 0.1
        else:
            horse_boost = 0
        
        return min(0.95, base_confidence + horse_boost)
    
    def _fallback_analysis(self, pdf_file):
        """Fallback analysis using PyPDF2"""
        try:
            pdf_file.seek(0)  # Reset file pointer
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            analysis = self._initialize_analysis(pdf_file)
            
            full_text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                full_text += page_text + "\n"
                self._analyze_page_content(page_text, analysis['pages_analyzed'], analysis)
            
            analysis['full_text'] = full_text
            analysis['detected_game'] = self._determine_game_type(analysis)
            analysis['confidence'] = self._calculate_confidence(analysis)
            
            return analysis
        except Exception as e:
            st.error(f"❌ Fallback analysis also failed: {e}")
            return self._get_empty_analysis()

# ========== STREAMLIT APPLICATION ==========
def main():
    st.set_page_config(
        page_title="QUANTUM QUINTE AI - PMUB Expert",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-title {
        font-size: 3.5rem;
        background: linear-gradient(45deg, #FF6B00, #FF0000, #FF0080, #FF00FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .pmub-card {
        background: rgba(255,107,0,0.1);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #FF6B00;
    }
    .confidence-high { color: #00FF00; font-weight: bold; }
    .confidence-medium { color: #FFFF00; font-weight: bold; }
    .confidence-low { color: #FF0000; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-title">
    🎯 QUANTUM QUINTE AI
    </div>
    <div style="text-align: center; color: #888; margin-bottom: 3rem;">
    Expert System for PMUB Games: Tiercé • Quarté • Quarté+1 • Quinté • Quinté+1
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = PMUBAnalyzer()
        st.session_state.current_analysis = None
        st.session_state.expert_system = PMUBExpertSystem()
    
    # Sidebar - PMUB Knowledge Base
    with st.sidebar:
        st.markdown("## 📚 PMUB KNOWLEDGE BASE")
        
        selected_game = st.selectbox(
            "Learn about PMUB Games:",
            list(st.session_state.expert_system.game_definitions.keys())
        )
        
        game_info = st.session_state.expert_system.game_definitions[selected_game]
        st.markdown(f"### {selected_game}")
        st.write(f"**Description:** {game_info['description']}")
        st.write(f"**Chevaux requis:** {game_info['horses_required']}")
        st.write(f"**Types de pari:** {', '.join(game_info['bet_types'])}")
        st.write(f"**Champ typique:** {game_info['typical_field']}")
        
        st.markdown("---")
        st.markdown("### 🎯 Detection Status")
        if st.session_state.current_analysis:
            analysis = st.session_state.current_analysis
            st.metric("Game Detected", analysis['detected_game'])
            confidence = analysis['confidence']
            confidence_class = "confidence-high" if confidence > 0.7 else "confidence-medium" if confidence > 0.4 else "confidence-low"
            st.markdown(f"**Confidence:** <span class='{confidence_class}'>{confidence:.0%}</span>", unsafe_allow_html=True)
            st.metric("Horses Found", analysis['total_horses_detected'])
            st.metric("Pages Analyzed", analysis['pages_analyzed'])
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📁 Upload PMUB PDF Document")
        uploaded_file = st.file_uploader(
            "Choose your PMUB PDF file", 
            type=['pdf'],
            help="Upload Tiercé, Quarté, Quarté+1, Quinté, or Quinté+1 PDF"
        )
        
        if uploaded_file:
            if uploaded_file != st.session_state.get('last_uploaded_file'):
                st.session_state.last_uploaded_file = uploaded_file
                
                with st.spinner("🔍 Performing expert PMUB analysis..."):
                    analysis = st.session_state.analyzer.analyze_pdf(uploaded_file)
                    st.session_state.current_analysis = analysis
                
                display_expert_analysis(analysis)
    
    with col2:
        st.markdown("## 🏇 PMUB Quick Guide")
        st.markdown("""
        <div class="pmub-card">
        **TIERCÉ**: 3 premiers chevaux dans l'ordre  
        **QUARTÉ**: 4 premiers chevaux  
        **QUARTÉ+1**: 4 premiers + 1 cheval  
        **QUINTÉ**: 5 premiers chevaux  
        **QUINTÉ+1**: 5 premiers + 1 cheval
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Detection Logic")
        st.write("""
        - **Direct Title Match**: Highest confidence
        - **Horse Count Analysis**: Fallback method  
        - **Bet Type Context**: Supporting evidence
        - **French Pattern Recognition**: Language-specific
        """)

def display_expert_analysis(analysis):
    """Display comprehensive analysis results"""
    st.success("✅ Expert PMUB analysis completed!")
    
    # Main results header
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        confidence = analysis['confidence']
        if confidence > 0.7:
            icon = "🟢"
        elif confidence > 0.4:
            icon = "🟡"
        else:
            icon = "🔴"
        st.metric("Confidence", f"{icon} {confidence:.0%}")
    
    with col2:
        st.metric("Game Type", analysis['detected_game'])
    
    with col3:
        st.metric("Horses Found", analysis['total_horses_detected'])
    
    with col4:
        st.metric("Pages", analysis['pages_analyzed'])
    
    # Detailed analysis sections
    with st.expander("🔍 DETECTION EVIDENCE & LOGIC", expanded=True):
        # Game evidence scores
        st.subheader("🎯 Game Detection Scores")
        evidence_df = pd.DataFrame([
            {'Game': game, 'Score': score} 
            for game, score in analysis['game_evidence'].items()
            if score > 0
        ])
        if not evidence_df.empty:
            st.dataframe(evidence_df.sort_values('Score', ascending=False), use_container_width=True)
        else:
            st.info("No direct game evidence found - using contextual analysis")
        
        # Detection details
        if analysis['detection_details']:
            st.subheader("📝 Detection Details")
            for detail in analysis['detection_details'][:15]:  # Limit display
                st.write(f"**{detail['game']}** (+{detail['weight']}): {detail['text']}")
    
    # Horse data section
    with st.expander("🐎 EXTRACTED HORSE DATA", expanded=True):
        if analysis['detected_horses']:
            horses_df = pd.DataFrame([{
                'Numéro': h['number'],
                'Nom': h['name'],
                'Détails': h['details'],
                'Confiance': h['extraction_confidence']
            } for h in analysis['detected_horses']])
            st.dataframe(horses_df.sort_values('Numéro'), use_container_width=True)
            
            # Horse count analysis
            st.subheader("📊 Horse Count Analysis")
            expected_horses = st.session_state.expert_system.game_definitions[analysis['detected_game']]['horses_required']
            st.write(f"**Detected:** {analysis['total_horses_detected']} horses")
            st.write(f"**Expected for {analysis['detected_game']}:** {expected_horses} horses")
            
            if analysis['total_horses_detected'] >= expected_horses:
                st.success("✅ Sufficient horses detected for this game type")
            else:
                st.warning("⚠️ Low horse count - detection confidence reduced")
        else:
            st.warning("No horse data extracted - check PDF quality")
    
    # Bet types found
    if analysis['bet_types_found']:
        with st.expander("💰 BET TYPES DETECTED"):
            bet_counter = Counter(analysis['bet_types_found'])
            for bet_type, count in bet_counter.items():
                st.write(f"**{bet_type}**: {count} mentions")
    
    # Action section
    st.markdown("---")
    st.markdown("## 🎯 READY FOR COMBINATION GENERATION")
    
    if st.button("🚀 GENERATE INTELLIGENT COMBINATIONS", type="primary", use_container_width=True):
        generate_pmub_combinations(analysis)

def generate_pmub_combinations(analysis):
    """Generate PMUB-specific combinations"""
    game_type = analysis['detected_game']
    horse_count = analysis['total_horses_detected']
    
    st.success(f"🎯 Generating {game_type} combinations with {horse_count} detected horses!")
    
    # Display game-specific information
    game_info = st.session_state.expert_system.game_definitions[game_type]
    st.markdown(f"### {game_type} Configuration")
    st.write(f"**Description:** {game_info['description']}")
    st.write(f"**Required horses:** {game_info['horses_required']}")
    st.write(f"**Bet types available:** {', '.join(game_info['bet_types'])}")
    
    # Combination generation logic would go here
    st.info("🔧 Combination engine would now process the detected horses and generate optimized bets...")
    
    # Example output
    if analysis['detected_horses']:
        horse_numbers = [h['number'] for h in analysis['detected_horses']]
        st.write(f"**Detected horse numbers:** {sorted(horse_numbers)}")
        st.write("**Sample combinations would be generated based on:**")
        st.write("- Horse performance data")
        st.write("- Track conditions")
        st.write("- Historical patterns")
        st.write("- Expert handicapping rules")

if __name__ == "__main__":
    main()
