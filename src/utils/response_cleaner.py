"""Response cleaning and formatting utilities"""

from typing import Optional


class ResponseCleaner:
    """Clean and format agent responses"""
    
    PREFIXES_TO_REMOVE = [
        "extracted answer:",
        "the answer is:",
        "answer:",
        "result:",
        "output:",
        "final answer:"
    ]
    
    CODE_PATTERNS = [
        "you can use:",
        "use the following command:",
        "you would need to",
        "however, i cannot perform",
        "once the dataframes are available:",
        ".groupby(",
        ".sum(",
        "holdings_df",
        "trades_df",
        "dfs[0]",
        "dfs[1]"
    ]
    
    @classmethod
    def clean(cls, response_text: str) -> str:
        """
        Clean response to remove duplicates, code snippets, and formatting issues.
        
        Args:
            response_text: Raw response from agent
            
        Returns:
            Cleaned response text
        """
        if not response_text:
            return ""
        
        lines = response_text.strip().split('\n')
        cleaned_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_lower = line.lower()
            
            # Skip lines with code patterns
            if any(pattern in line_lower for pattern in cls.CODE_PATTERNS):
                continue
            
            # Skip lines that are code
            if '`' in line or 'df[' in line or '.shape[0]' in line:
                continue
            
            # Remove prefixes
            for prefix in cls.PREFIXES_TO_REMOVE:
                if line_lower.startswith(prefix):
                    line = line[len(prefix):].strip()
                    line_lower = line.lower()
            
            # Only add unique lines
            if line_lower not in seen_lines and line and len(line) > 10:
                seen_lines.add(line_lower)
                cleaned_lines.append(line)
        
        # Return first meaningful line
        if cleaned_lines:
            return cleaned_lines[0]
        
        # Check if original contains code
        response_lower = response_text.lower()
        if any(pattern in response_lower for pattern in cls.CODE_PATTERNS):
            return "Sorry can not find the answer"
        
        return response_text.strip()
    
    @classmethod
    def contains_code(cls, text: str) -> bool:
        """Check if text contains code snippets"""
        return any(pattern in text.lower() for pattern in cls.CODE_PATTERNS)