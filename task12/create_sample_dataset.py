import os
import random

# Create directories
os.makedirs('data/enron-spam/ham', exist_ok=True)
os.makedirs('data/enron-spam/spam', exist_ok=True)

print("Creating sample dataset...")

# Ham email templates
ham_templates = [
    """Subject: Meeting reminder
    
Hi team,

Just a reminder that we have a meeting scheduled for tomorrow at 10 AM.
Please bring your project updates and be prepared to discuss the quarterly goals.

Best regards,
John""",

    """Subject: Project update
    
Hello everyone,

I wanted to share some updates on the current project:
- Phase 1 is complete
- Phase 2 will start next week
- We're on track to meet our deadline

Let me know if you have any questions.

Thanks,
Sarah""",

    """Subject: Weekly report
    
Hi all,

Please find attached the weekly report for our department.
The numbers look good and we're making progress on all fronts.

See you at the team lunch on Friday.

Regards,
Michael""",

    """Subject: Document review request
    
Hello,

Could you please review the attached document when you have a moment?
I need your feedback by end of day tomorrow.

Thank you,
Emma""",

    """Subject: Holiday schedule
    
Team,

Just a reminder that the office will be closed next Monday for the holiday.
Please make sure to complete any urgent tasks before the weekend.

Have a great holiday!
HR Department"""
]

# Spam email templates
spam_templates = [
    """Subject: URGENT: Your account needs verification
    
DEAR CUSTOMER,

YOUR ACCOUNT HAS BEEN FLAGGED FOR VERIFICATION. YOU MUST CLICK THE LINK BELOW IMMEDIATELY TO AVOID SUSPENSION.

www.secure-verification-link.com

ACT NOW!!! LIMITED TIME OFFER!!! CLICK HERE TO CLAIM YOUR PRIZE!!!

Best regards,
Account Team""",

    """Subject: Congratulations! You've won a free iPhone!
    
CONGRATULATIONS!!!

You have been selected as our lucky winner for a brand new iPhone!

To claim your FREE prize, simply:
1. Send your bank details
2. Pay a small processing fee of $9.99
3. Forward this email to 10 friends

LIMITED TIME OFFER! ACT NOW!!!

Best wishes,
Prize Department""",

    """Subject: MAKE MONEY FAST!!! Work from home opportunity!!!
    
$$$ EARN THOUSANDS FROM HOME $$$

Our revolutionary system allows you to make THOUSANDS of dollars with minimal effort!
Join our program today and start earning IMMEDIATELY!

100% GUARANTEED RESULTS or your money back!

Click here: www.get-rich-quick.com""",

    """Subject: Your Payment Has Been Processed
    
Dear Valued Customer,

We have processed your payment of $499.99 for Premium Membership.
If you did not authorize this transaction, please CLICK HERE to dispute.

You have 24 HOURS to respond or the charge will be PERMANENT!

Customer Service""",

    """Subject: Pharmacy Discount - 90% OFF All Medications
    
**DISCOUNT PHARMACY**

Buy all medications at 90% OFF regular price!
- No prescription needed!
- Fast shipping worldwide!
- 100% satisfaction guaranteed!

CLICK HERE TO ORDER NOW: www.discount-meds-online.com"""
]

# Create ham emails
for i in range(1, 201):
    template = random.choice(ham_templates)
    # Add some random variations to make emails unique
    variations = [
        f"\nP.S. Meeting room has been changed to {random.choice(['A', 'B', 'C'])}-{random.randint(100, 999)}.",
        f"\nP.S. Please respond by {random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])}.",
        f"\nP.S. The {random.choice(['quarterly', 'monthly', 'weekly'])} report is due soon.",
        "",
        f"\nP.S. Don't forget to bring your {random.choice(['laptop', 'notes', 'presentation'])}."
    ]
    
    email_content = template + random.choice(variations)
    
    with open(f'data/enron-spam/ham/ham_{i}.txt', 'w') as f:
        f.write(email_content)

# Create spam emails
for i in range(1, 201):
    template = random.choice(spam_templates)
    # Add some random variations to make emails unique
    variations = [
        f"\nOffer expires on {random.randint(1, 30)}/{random.randint(1, 12)}/{random.randint(2023, 2025)}!",
        f"\nUse code: {random.choice(['FREE', 'DISCOUNT', 'SPECIAL'])}{random.randint(100, 999)} at checkout!",
        f"\nThis offer is valid for the next {random.randint(1, 24)} hours only!",
        "",
        f"\nReply now to claim your {random.choice(['free gift', 'discount', 'bonus offer'])}!"
    ]
    
    email_content = template + random.choice(variations)
    
    with open(f'data/enron-spam/spam/spam_{i}.txt', 'w') as f:
        f.write(email_content)

print("Sample dataset created with 400 emails (200 ham, 200 spam)")