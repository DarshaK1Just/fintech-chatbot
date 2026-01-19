"""Input validation utilities"""


class QuestionValidator:
    """Validate user questions"""
    
    EXTERNAL_KEYWORDS = [
        "capital of", "what is bitcoin", "price of bitcoin", "current price",
        "stock price", "market price", "real-time", "live price",
        "what is", "who is", "when did", "history of", "explain"
    ]
    
    FINANCIAL_KEYWORDS = [
        "fund", "portfolio", "holding", "trade", "profit", "loss",
        "pl_ytd", "ytd", "transaction", "security", "quantity"
    ]
    
    @classmethod
    def validate(cls, question: str) -> bool:
        """
        Validate if question is appropriate for the dataset.
        
        Args:
            question: User's question
            
        Returns:
            True if valid, False otherwise
        """
        question_lower = question.lower()
        
        # Check for financial context
        has_financial_context = any(
            keyword in question_lower for keyword in cls.FINANCIAL_KEYWORDS
        )
        
        # Check for external knowledge questions
        if any(keyword in question_lower for keyword in cls.EXTERNAL_KEYWORDS):
            if not has_financial_context:
                # Might still be valid if asking about fund data
                if "fund" in question_lower or "portfolio" in question_lower:
                    return True
                return False
        
        return True