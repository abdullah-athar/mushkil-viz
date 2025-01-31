import csv
from datetime import datetime, timedelta
from faker import Faker
import random
import calendar

fake = Faker()

# Define categories and their associated merchants and amount ranges
CATEGORIES = {
    'Income': {
        'merchants': ['Freelance Client', 'Side Gig LLC'],
        'amount_range': (100, 1000),
        'is_income': True
    },
    "savings": {
        'merchants': ['Savings Account'],
        'amount_range': (100, 500),
        'is_income': False
    },
    'Groceries': {
        'merchants': ['Walmart', 'Costco', 'Whole Foods', 'Trader Joe\'s', 'Safeway'],
        'amount_range': (30, 200),
        'is_income': False
    },
    'Food & Dining': {
        'merchants': ['Starbucks', 'Local Restaurant', 'Chipotle', 'McDonald\'s', 'Subway'],
        'amount_range': (10, 100),
        'is_income': False
    },
    'Shopping': {
        'merchants': ['Amazon', 'Target', 'Best Buy', 'Apple Store', 'Nike'],
        'amount_range': (20, 300),
        'is_income': False
    },
    'Transportation': {
        'merchants': ['Shell', 'Uber', 'Lyft', 'Chevron', 'BP'],
        'amount_range': (20, 80),
        'is_income': False
    },
    'Entertainment': {
        'merchants': ['Netflix', 'Spotify', 'AMC Theaters', 'Disney+', 'HBO Max'],
        'amount_range': (10, 50),
        'is_income': False
    },
    'Bills': {
        'merchants': ['Utility Company', 'AT&T', 'Verizon', 'Water Corp', 'Internet Provider'],
        'amount_range': (50, 200),
        'is_income': False
    },
    'Housing': {
        'merchants': ['Landlord', 'Property Management', 'Mortgage Co.'],
        'amount_range': (800, 2500),
        'is_income': False
    },
    'Health': {
        'merchants': ['Pharmacy', 'CVS', 'Walgreens', 'Medical Center', 'Dental Clinic'],
        'amount_range': (20, 150),
        'is_income': False
    }
}

def get_last_day_of_month(date):
    return calendar.monthrange(date.year, date.month)[1]

def generate_monthly_salary(date):
    """Generate a fixed monthly salary transaction."""
    salary_amount = 3000.00  # Fixed monthly salary
    return [date.strftime('%Y-%m-%d'), f'{salary_amount:.2f}', 'Employer Inc', 'Income', 'Monthly Salary']

def generate_transaction(date, exclude_employer_income=False):
    """Generate a random transaction, optionally excluding employer income."""
    while True:
        # Randomly select a category
        category = random.choice(list(CATEGORIES.keys()))
        category_data = CATEGORIES[category]
        
        # Skip employer income if excluded
        if exclude_employer_income and category == 'Income':
            merchant = random.choice([m for m in category_data['merchants'] if m != 'Employer Inc'])
        else:
            merchant = random.choice(category_data['merchants'])
        
        # Skip if it's employer income when excluded
        if exclude_employer_income and merchant == 'Employer Inc':
            continue
            
        # Generate amount
        amount = round(random.uniform(*category_data['amount_range']), 2)
        if not category_data['is_income']:
            amount = -amount
        
        # Generate description
        if category == 'Income':
            description = 'Monthly Salary' if merchant == 'Employer Inc' else f'Payment from {merchant}'
        else:
            description = fake.sentence(nb_words=4)[:-1]  # Remove the period at the end
        
        return [date.strftime('%Y-%m-%d'), f'{amount:.2f}', merchant, category, description]

def generate_financial_data(num_days=180):
    transactions = []
    start_date = datetime.now() - timedelta(days=num_days)
    
    # Generate header
    header = ['date', 'amount', 'merchant', 'category', 'description']
    transactions.append(header)
    
    # Generate transactions for each day
    current_date = start_date
    while current_date <= datetime.now():
        # Add salary payment on the last day of each month
        if current_date.day == get_last_day_of_month(current_date):
            transactions.append(generate_monthly_salary(current_date))
        
        # Generate 1-5 regular transactions per day (excluding employer income)
        num_transactions = random.randint(1, 5)
        for _ in range(num_transactions):
            transaction = generate_transaction(current_date, exclude_employer_income=True)
            transactions.append(transaction)
        
        current_date += timedelta(days=1)
    
    return transactions

def save_to_csv(transactions, filename='financial_data.csv'):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(transactions)

if __name__ == '__main__':
    # Generate 90 days of financial data
    transactions = generate_financial_data(180)
    save_to_csv(transactions, 'examples/financial_data.csv')
    print("Generated new financial_data.csv with dummy transactions.") 