# weak_labeler.py
import re
import json
from typing import List, Dict
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int

class WeakLabeler:
    """Weak labeling system for Nepali/English documents - COMPLETE VERSION"""
    
    def __init__(self):
        # Regex patterns for different entities
        self.patterns = {
            # Citizenship numbers: Handle OCR errors like ? and mixed numbers
            'CITIZENSHIP_NUMBER': [
                r'प्रजं\.?\s*([\d०-९\-\.\?]+)',  # ना.प्रजं. 30-0?-08-0?४.0
                r'प्रजा॰\.?\s*([\d०-९\-\.\?]+)',  # ना. प्रजा॰ 30-0-08-04
                r'नाःप्रःनं\.?\s*([\d०-९\-\.\?\s]+)',  # नाःप्रःनं. ; २७-०? ४-०४४८२
                r'नापप्रग्न.*?:\s*([\d०-९\-\.\?]+)',  # नापप्रग्न% : ३४-०१-७५-०१६३७
                r'\b[\d०-९]{1,2}[-\s\?]+[\d०-९]{1,2}[-\s\?]+[\d०-९]{1,2}[-\s\?]+[\d०-९]{4,5}\b',
                r'प्रजं[^\d]*([\d०-९].*?[\d०-९])',
            ],
            
            'CITIZENSHIP_NUMBER_EN': [
                r'Citizenship\s+[A-Za-z]+\s+No\.?\s*[:\-]\s*([\d\s\-]+)',  # Citizenship Certificete No. - 052028 1 33919
                r'Citizenship\s+Certificate\s+No\.:\s*([\d\s\-]+)',  # Citizenship Certificate No.: 28-01-72-00911
                r'\b\d{2}\s*[\-\s]\s*\d{2}\s*[\-\s]\s*\d{2}\s*[\-\s]\s*\d{4,5}\b',
                r'No\.:\s*([\d\-]+)',
            ],
            
            # Names: More specific patterns to avoid grabbing too much
            'NAME': [
                r'नाम\s+थरः\s+([^\n:।लिङ्ग]{2,25})(?=\s+लिङ्ग|\s+जन्म|\s*[।\n]|$)',  # Stop before लिङ्ग or जन्म
                r'नाम\s+थर\s+([^\n:।लिङ्ग]{2,25})(?=\s+लिङ्ग|\s+जन्म|\s*[।\n]|$)',
                r'नामपाः\s+([^\n:।लिङ्ग]{2,25})(?=\s+लिङ्ग|\s+जन्म|\s*[。\n]|$)',
                r'नाम\s+यर\s+([^\n:।लिङ्ग]{2,25})(?=\s+लिङ्ग|\s+जन्म|\s*[।\n]|$)',
                r'(?<=नाम थरः\s)([^लिङ्ग]{2,30})(?=\s+लिङ्ग)',  # Specific for Dawa case
            ],
            
            'NAME_EN': [
                r'Full\s+Name\s*[\(:]?\s*in\s+block\)?\s*:\s*([A-Z][A-Z\s]{2,30}[A-Z])(?=\s+Sex|\s+Date|$)',  # Stop before Sex or Date
                r'Full\s+Name\.:\s*([A-Z][A-Z\s]{2,30}[A-Z])(?=\s+Sex|\s+Date|$)',
                r'Name\s*:\s*([A-Z][A-Z\s]{2,30}[A-Z])(?=\s+Sex|\s+Date|$)',
                r'Full Name\.:\s*([^\n]+?)(?=\s+Sex|\s+Date|\n|$)',
            ],
            
            # Gender: Improved patterns
            'GENDER': [
                r'लिङ्ग\s+([^\s\n:]{3,8})(?=\s+जन्म|\s+महिला|\s+पुरुष|\s*[।\n]|$)',  # Specific follow words
                r'लिङ्ग\s*:\s*([^\s\n:]{3,8})(?=\s+जन्म|\s+महिला|\s+पुरुष|\s*[।\n]|$)',
                r'\b(महिला|पुरुष|पुरुंष|पुरुब|स्त्री|अन्य|निङ्ग)\b(?=\s+जन्म|\s+जन्मम्थानः|\s*[।\n]|$)',
                r'(?<=लिङ्ग)[\s:]*([^\s\n:]{3,8})(?=\s+जन्म|\s*[।\n]|$)',  # Catch anything after लिङ्ग
            ],
            
            'GENDER_EN': [
                r'Sex\s*:\s*([A-Za-z\.]+)(?=\s+Date|\s+Full|\s*[\.\n]|$)',  # Allow N. (single letter with dot)
                r'Sex\s+([A-Za-z\.]+)(?=\s+Date|\s+Full|\s*[\.\n]|$)',
                r'\b(Male|Female|Other|N\.|M\.|F\.)\b(?=\s+Date|\s+Full|\s*[\.\n]|$)',
                r'Sex\s*:\s*([^\n]{1,10})(?=\s+Date|\s+Full|\n|$)',
            ],
            
            # Dates: Better patterns
            'DATE': [
                r'सालः\s*([०-९]{4})(?=\s+महिनाः|\s*[।\n]|$)',
                r'महिनाः\s*([०-९]{1,2})(?=\s+गतेः|\s*[।\n]|$)',
                r'गतेः\s*([०-९]{1,2})(?=\s+[^\s]|\s*[।\n]|$)',
                r'मितिः\s*([०-९]{4})(?=\s+महिनाः|\s*[।\n]|$)',
                r'\b[०-९]{4}\b',  # Catch any 4-digit Nepali year
                r'सालः\s*([०-९]+)',
                r'महिनाः\s*([०-९]+)',
                r'गतेः\s*([०-९]+)',
            ],
            
            'DATE_EN': [
                r'Year\s*:\s*(\d{4})(?=\s+Month|\s+Day|\s*[\.\n]|$)',
                r'Month\s*:\s*([A-Za-z\d]{2,})(?=\s+Day|\s*[\.\n]|$)',  # Allow digits (99)
                r'Day\s*:\s*(\d{1,2})(?=\s+Birth|\s*[\.\n]|$)',
                r'Date of Birth.*?Year[:\s]*(\d{4})',
                r'Date of Birth.*?Month[:\s]*([A-Za-z\d]{2,})',
                r'Date of Birth.*?Day[:\s]*(\d{1,2})',
                r'\b\d{4}\b',  # Catch any 4-digit year
                r'\b\d{1,2}\b',  # Catch any 1-2 digit day/month
                r'Year[:\s]*(\d{4})',
                r'Month[:\s]*([A-Za-z\d]+)',
                r'Day[:\s]*(\d{1,2})',
            ],
            
            # District: More precise
            'DISTRICT': [
                r'जिल्ला\s*:\s*([^\n:।]{3,20})(?=\s+[नगाः]|\s*[।\n]|$)',  # Stop before next field
                r'जिल्ला\s+([^\n:।]{3,20})(?=\s*[:।\n]|$)',
                r'जिल्ला[^\n]{0,20}:\s*([^\n:।]{3,20})',
            ],
            
            # Ward: Handle OCR variations
            'WARD': [
                r'वडा\s*नं\.?\s*:\s*([०-९\d]+)(?=\s+[^\s]|\s*[।\n]|$)',
                r'वडा\s*नं\.?\s*([०-९\d]+)(?=\s+[^\s]|\s*[।\n]|$)',
                r'वडा\s*:\s*([०-९\d]+)(?=\s+[^\s]|\s*[।\n]|$)',
                r'बडा\s*न\.?\s*([०-९\d]+)(?=\s+[^\s]|\s*[।\n]|$)',  # OCR misreads व as ब
                r'वडा\s*न\.?\s*:\s*([०-९\d]+)',
                r'यडा\s*न\.?\s*([०-९\d]+)',  # OCR misreads व as य
            ],
            
            'WARD_EN': [
                r'Ward\s+No\.\s*:\s*(\d+)(?=\s+[A-Z]|\s*[\.\n]|$)',
                r'Ward\s+No\.\s*(\d+)(?=\s+[A-Z]|\s*[\.\n]|$)',
                r'Ward\s+No\.\s*:\s*(\d+)',
            ],
            
            # Municipality
            'MUNICIPALITY': [
                r'नगरपालिका\s*:\s*([^\n:।]{3,30})',
                r'नःपा\.?\s*:\s*([^\n:।]{3,30})',
                r'गा\.वि\.स\.\s*:\s*([^\n:।]{3,30})',
                r'गाःवि[^\n]{0,10}:\s*([^\n:।]{3,30})',
            ],
            
            'MUNICIPALITY_EN': [
                r'Municipality\s*:\s*([^\n:]{3,30})',
                r'VDC\s*:\s*([^\n:]{3,30})',
                r'Sub-Metropolitan\s*:\s*([^\n:]{3,30})',
            ],
        }
        
        # Gazetteers for common values
        self.gazetteers = {
            'GENDER': ['पुरुष', 'महिला', 'पुंष', 'स्त्री', 'अन्य', 'निङ्ग'],
            'GENDER_EN': ['Male', 'Female', 'Other', 'N.', 'M.', 'F.', 'N'],
            'MONTH_EN': ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                        'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'APRIL', 'OCTOBER', '99'],
            'DISTRICT': [
                'काभ्रेपलाञ्चोक', 'काठमाडौं', 'म्याग्दी', 'सप्तरी', 
                'उदयपुर', 'सिन्धुपाल्चोक', 'ललितपुर', 'मकवानपुर',
                'मोरङ', 'गुल्मी', 'बागलुङ', 'कास्की', 'पर्सा',
                'सर्लाही', 'गाःदि', 'बागलुड', 'सिन्धुपाल्चोक', 'गुल्मी',
                'सप्तरी', 'उदयपुर', 'बागलुङ', 'ललितपुर', 'मकवानपुर'
            ],
        }
        
        # Common OCR error mappings for cleaning
        self.ocr_corrections = {
            'प्रजं': 'प्रजा॰',
            'नःपाः': 'नगरपालिका',
            'नःपा': 'नगरपालिका',
            'यडा': 'वडा',
            'बडा': 'वडा',
            'जिल्ना': 'जिल्ला',
            'निङ्ग': 'लिङ्ग',
            'गाःवि': 'गा.वि.स.',
            'गाःपि': 'गा.वि.स.',
        }
    
    def label_text(self, text: str, language: str = "auto") -> List[Entity]:
        """Label entities in text using weak supervision"""
        entities = []
        
        # Auto-detect language
        if language == "auto":
            language = self._detect_language(text)
        
        # Clean text slightly for better matching (but keep original for positions)
        cleaned_text = self._clean_ocr_text(text)
        
        # Find entities using regex patterns
        for label, patterns in self.patterns.items():
            # Skip patterns for wrong language
            if language == "ne" and label.endswith("_EN"):
                continue
            if language == "en" and not label.endswith("_EN"):
                # But keep base labels (like DATE) for English too
                if label not in ['DATE']:  # DATE works for both
                    continue
            
            for pattern in patterns:
                try:
                    for match in re.finditer(pattern, cleaned_text, re.IGNORECASE):
                        # Extract the actual entity text
                        if match.groups():
                            entity_text = match.group(1)
                        else:
                            entity_text = match.group(0)
                        
                        if entity_text:
                            # Clean the text
                            entity_text = entity_text.strip(' :.,;।\n\t')
                            
                            # Skip if too short or invalid
                            if len(entity_text) < 2:
                                continue
                                
                            # Validate against gazetteers if applicable
                            if self._is_valid_entity(label, entity_text, language):
                                # Map position back to original text
                                # For simplicity, use same positions (should be close)
                                entities.append(Entity(
                                    text=entity_text,
                                    label=label,
                                    start=match.start(),
                                    end=match.end()
                                ))
                except Exception as e:
                    continue
        
        # Remove overlapping entities and clean up
        deduplicated = self._deduplicate_entities(entities)
        
        # Post-process: Fix common issues
        final_entities = self._post_process_entities(deduplicated, language)
        
        return final_entities
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean common OCR errors to improve pattern matching"""
        cleaned = text
        for wrong, correct in self.ocr_corrections.items():
            cleaned = cleaned.replace(wrong, correct)
        return cleaned
    
    def _post_process_entities(self, entities: List[Entity], language: str) -> List[Entity]:
        """Fix common entity extraction issues"""
        result = []
        
        for entity in entities:
            # Fix NAME entities that include extra words
            if entity.label == 'NAME':
                text = entity.text
                # Remove common extra words
                for extra in ['लिङ्ग', 'महिला', 'पुरुष', 'जन्म', 'स्थान', 'जिल्ला', 'जन्मम्थानः']:
                    if extra in text:
                        text = text.split(extra)[0].strip()
                # Also split on ":" if present
                if ':' in text:
                    text = text.split(':')[0].strip()
                if text and len(text) >= 2:
                    entity.text = text
                    result.append(entity)
            
            # Fix DISTRICT entities that include extra
            elif entity.label == 'DISTRICT':
                text = entity.text
                # Remove municipality indicators and garbage
                for extra in ['नगरपालिका', 'नःपाः', 'गा.वि.स.', 'गाःवि', 'गाभयिः', 'सः', ':']:
                    if extra in text:
                        text = text.replace(extra, '').strip()
                # Take only first word if multiple
                if ' ' in text:
                    words = text.split()
                    if len(words) > 1:
                        text = words[0]  # Take just the district name
                if text and len(text) >= 2:
                    entity.text = text
                    result.append(entity)
            
            # Fix GENDER entities
            elif 'GENDER' in entity.label:
                text = entity.text
                # Clean up
                text = text.strip(' :.')
                if text in ['N', 'M', 'F']:
                    text = text + '.'  # Add dot for consistency
                if text:
                    entity.text = text
                    result.append(entity)
            
            else:
                # Keep other entities as-is
                result.append(entity)
        
        return result
    
    def _is_valid_entity(self, label: str, text: str, language: str) -> bool:
        """Validate extracted entity"""
        # Skip empty or very short text
        if not text or len(text.strip()) < 2:
            return False
        
        # Clean the text for validation
        clean_text = text.strip(' :.,;।')
        
        # Check gazetteers
        if label in self.gazetteers:
            text_upper = clean_text.upper()
            for valid in self.gazetteers[label]:
                if valid.upper() == text_upper or text_upper == valid.upper():
                    return True
                # Check for partial matches for districts
                if label == 'DISTRICT' and len(clean_text) >= 3:
                    # If text is part of a known district
                    for district in self.gazetteers['DISTRICT']:
                        if clean_text in district or district in clean_text:
                            return True
        
        # Citizenship number validation
        if 'CITIZENSHIP' in label:
            # Check if it looks like a citizenship number
            if re.search(r'[\d०-९].*?[\-\s].*?[\d०-९].*?[\-\s].*?[\d०-९]', clean_text):
                return True
            # Also accept if it has numbers and dashes
            if any(c in '0123456789०१२३४५६७८९-' for c in clean_text):
                return True
        
        # Date validation
        if 'DATE' in label:
            if language == "en":
                # Check for year
                if re.search(r'\b(19|20)\d{2}\b', clean_text):
                    return True
                # Check for month name
                if any(month in clean_text.upper() for month in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                                                               'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']):
                    return True
                # Check for day (1-31)
                if re.search(r'\b([1-9]|[12][0-9]|3[01])\b', clean_text):
                    return True
            else:
                # Nepali date has Devanagari numbers
                if re.search(r'[०१२३४५६७८९]+', clean_text):
                    return True
        
        # Name validation
        if 'NAME' in label:
            # Should be reasonable length and not contain field labels
            invalid_words = ['लिङ्ग', 'जन्म', 'स्थान', 'जिल्ला', 'Sex', 'Date', 'Birth', 'थरः']
            if any(word in clean_text for word in invalid_words):
                return False
            # Should have at least 2 characters and not be a number
            if len(clean_text) >= 2 and not re.search(r'^\d+$', clean_text):
                return True
        
        # District/Municipality - accept if reasonable length
        if label in ['DISTRICT', 'MUNICIPALITY']:
            return 2 <= len(clean_text) <= 30
        
        # Ward number
        if 'WARD' in label:
            return clean_text.isdigit() or all(c in '०१२३४५६७८९' for c in clean_text)
        
        # Gender should be in gazetteer (already checked)
        if 'GENDER' in label:
            return True  # Already validated by gazetteer check
        
        # By default, accept if reasonable
        return 2 <= len(clean_text) <= 100
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        nepali_chars = set('अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहािीुूृेैोौंःँ')
        
        # Take first 500 chars for efficiency
        sample = text[:500]
        
        has_nepali = any(char in nepali_chars for char in sample)
        has_english = any('A' <= char <= 'Z' or 'a' <= char <= 'z' for char in sample)
        
        if has_nepali and not has_english:
            return "ne"
        elif has_english and not has_nepali:
            return "en"
        else:
            # Count characters
            nepali_count = sum(1 for char in sample if char in nepali_chars)
            english_count = sum(1 for char in sample if 'A' <= char <= 'Z' or 'a' <= char <= 'z')
            
            if nepali_count > english_count:
                return "ne"
            else:
                return "en"
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities, keep the most specific"""
        if not entities:
            return []
        
        # Sort by start position
        entities.sort(key=lambda x: x.start)
        
        filtered = []
        for entity in entities:
            if not filtered:
                filtered.append(entity)
            else:
                last = filtered[-1]
                # Check for overlap
                overlap = entity.start < last.end
                
                if not overlap:
                    filtered.append(entity)
                else:
                    # Choose based on priority
                    priority = {
                        'CITIZENSHIP_NUMBER': 10,
                        'CITIZENSHIP_NUMBER_EN': 10,
                        'NAME': 9,
                        'NAME_EN': 9,
                        'GENDER': 8,
                        'GENDER_EN': 8,
                        'DATE': 7,
                        'DATE_EN': 7,
                        'DISTRICT': 6,
                        'WARD': 5,
                        'MUNICIPALITY': 4,
                    }
                    
                    entity_priority = priority.get(entity.label, 0)
                    last_priority = priority.get(last.label, 0)
                    
                    if entity_priority > last_priority:
                        filtered[-1] = entity
                    elif entity_priority == last_priority:
                        # Keep shorter (more specific) entity
                        if len(entity.text) < len(last.text):
                            filtered[-1] = entity
                    # else keep the last one
        
        return filtered

def visualize_entities(text: str, entities: List[Entity]):
    """Create a visualization of entities in text"""
    if not text or not entities:
        return text
    
    # Sort entities by start position in reverse order
    sorted_entities = sorted(entities, key=lambda x: x.start, reverse=True)
    
    # Insert markers
    result = text
    for entity in sorted_entities:
        if entity.start < len(result) and entity.end <= len(result):
            # Short label for display
            short_label = entity.label.replace('_EN', '').replace('_NUMBER', '')
            tagged = f"[{entity.text}]({short_label})"
            result = result[:entity.start] + tagged + result[entity.end:]
    
    return result