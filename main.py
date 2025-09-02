"""
E-COMMERCE SALES ANALYTICS SYSTEM
=================================
Professional data analysis project demonstrating:
- Data generation and processing
- Statistical analysis and insights  
- Business intelligence and visualization
- Automated reporting and exports

Author: Harish Prabhu P
Purpose: Data Analyst Portfolio Project
GitHub: github.com/harishprabhu/ecommerce-sales-analytics
"""

# =============================================================================
# IMPORTS - Core Data Science Libraries
# =============================================================================
import pandas as pd              # Data manipulation (think Excel on steroids)
import numpy as np              # Mathematical operations and statistics
import matplotlib.pyplot as plt # Chart creation and visualization
import seaborn as sns           # Professional statistical visualizations
from datetime import datetime, timedelta  # Date and time handling
import warnings
import os

# Configure for clean output
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🚀 E-COMMERCE SALES ANALYTICS SYSTEM")
print("=" * 60)
print("🎯 Demonstrating Professional Data Analyst Skills")
print("📊 Processing → Analysis → Visualization → Business Insights")
print("=" * 60)

# =============================================================================
# SETUP PROJECT STRUCTURE
# =============================================================================
def create_project_folders():
    """Create organized folder structure for professional project management"""
    folders = ['data', 'outputs', 'outputs/charts', 'outputs/reports']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"📁 Folder ready: {folder}")

# =============================================================================
# DATA GENERATOR - Creates Realistic Business Data
# =============================================================================
class BusinessDataGenerator:
    """
    Generates realistic e-commerce transaction data
    Simulates what you'd receive from company databases
    """
    
    def __init__(self, num_transactions=10000):
        self.num_transactions = num_transactions
        np.random.seed(42)  # Ensures reproducible results
        print(f"\n🔧 Initializing data generator for {num_transactions:,} transactions")
    
    def generate_ecommerce_data(self):
        """Create realistic business transaction dataset"""
        
        # Business categories with realistic pricing
        categories = {
            'Electronics': {'avg_price': 450, 'std': 180, 'weight': 0.35},
            'Clothing': {'avg_price': 65, 'std': 25, 'weight': 0.25}, 
            'Home & Garden': {'avg_price': 95, 'std': 40, 'weight': 0.20},
            'Sports': {'avg_price': 55, 'std': 20, 'weight': 0.12},
            'Books': {'avg_price': 22, 'std': 8, 'weight': 0.08}
        }
        
        # Customer segments (business typically uses these)
        segments = ['Premium', 'Regular', 'Budget']
        segment_weights = [0.15, 0.60, 0.25]  # Realistic distribution
        
        # Generate realistic date range (18 months of data)
        end_date = datetime.now() - timedelta(days=30)
        start_date = end_date - timedelta(days=540)
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        print("📊 Generating realistic business transactions...")
        
        transactions = []
        for i in range(self.num_transactions):
            
            # Select category based on business weights
            cat_names = list(categories.keys())
            cat_weights = [categories[c]['weight'] for c in cat_names]
            category = np.random.choice(cat_names, p=cat_weights)
            
            # Customer segment affects buying behavior
            segment = np.random.choice(segments, p=segment_weights)
            
            # Realistic pricing based on category and segment
            base_price = np.random.normal(
                categories[category]['avg_price'], 
                categories[category]['std']
            )
            
            # Segment pricing multipliers
            multipliers = {'Premium': 1.8, 'Regular': 1.0, 'Budget': 0.65}
            unit_price = max(5, base_price * multipliers[segment])
            
            # Quantity (most customers buy 1-2 items)
            quantity = np.random.choice([1, 2, 3, 4], p=[0.60, 0.25, 0.12, 0.03])
            
            # Calculate revenue
            total_revenue = unit_price * quantity
            
            # Add seasonal effects (realistic business patterns)
            transaction_date = np.random.choice(date_range)
            month = transaction_date.month
            if month in [11, 12]:  # Holiday boost
                total_revenue *= np.random.uniform(1.15, 1.40)
            elif month in [6, 7]:  # Summer sales
                total_revenue *= np.random.uniform(1.05, 1.20)
            
            # Create transaction record
            transaction = {
                'transaction_id': f'T{i+1:06d}',
                'date': transaction_date.date(),
                'customer_id': f'C{np.random.randint(1, 2000):04d}',
                'category': category,
                'customer_segment': segment,
                'quantity': quantity,
                'unit_price': round(unit_price, 2),
                'total_revenue': round(total_revenue, 2),
                'region': np.random.choice(['North', 'South', 'East', 'West', 'Central']),
                'day_of_week': transaction_date.strftime('%A'),
                'month': month,
                'quarter': f"Q{(month-1)//3 + 1}"
            }
            
            transactions.append(transaction)
        
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Add derived business metrics
        df['date'] = pd.to_datetime(df['date'])
        df['month_year'] = df['date'].dt.to_period('M')
        df['is_weekend'] = df['date'].dt.weekday.isin([5, 6])
        
        print(f"✅ Generated {len(df):,} transactions")
        print(f"📅 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"💰 Total revenue: ${df['total_revenue'].sum():,.2f}")
        
        return df

# =============================================================================
# ANALYTICS ENGINE - Core Business Analysis
# =============================================================================
class ECommerceAnalytics:
    """Professional analytics engine for business intelligence"""
    
    def __init__(self, data):
        self.df = data
        self.insights = {}
        print(f"\n📈 Analytics engine loaded with {len(self.df):,} records")
    
    def business_overview(self):
        """Generate executive summary metrics"""
        print("\n" + "="*50)
        print("💼 EXECUTIVE BUSINESS SUMMARY")
        print("="*50)
        
        # Key Performance Indicators
        total_revenue = self.df['total_revenue'].sum()
        total_transactions = len(self.df)
        unique_customers = self.df['customer_id'].nunique()
        avg_order_value = self.df['total_revenue'].mean()
        
        print(f"💰 Total Revenue: ${total_revenue:,.2f}")
        print(f"🛒 Total Transactions: {total_transactions:,}")
        print(f"👥 Unique Customers: {unique_customers:,}")
        print(f"📊 Average Order Value: ${avg_order_value:.2f}")
        print(f"💎 Revenue per Customer: ${total_revenue/unique_customers:.2f}")
        
        # Store for reporting
        self.insights['overview'] = {
            'total_revenue': total_revenue,
            'total_transactions': total_transactions,
            'avg_order_value': avg_order_value,
            'unique_customers': unique_customers
        }
        
        return self.insights['overview']
    
    def revenue_analysis(self):
        """Analyze revenue patterns and growth trends"""
        print("\n" + "="*50)
        print("📈 REVENUE PERFORMANCE ANALYSIS")
        print("="*50)
        
        # Monthly revenue trends
        monthly_revenue = self.df.groupby('month_year')['total_revenue'].sum()
        monthly_growth = monthly_revenue.pct_change().mean() * 100
        
        print("📅 Monthly Revenue Performance:")
        print(monthly_revenue.tail(6).to_string())
        print(f"\n📈 Average Monthly Growth: {monthly_growth:.1f}%")
        
        # Category performance
        category_performance = self.df.groupby('category').agg({
            'total_revenue': ['sum', 'count', 'mean'],
            'customer_id': 'nunique'
        }).round(2)
        
        category_performance.columns = ['total_revenue', 'transactions', 'avg_revenue', 'customers']
        category_performance = category_performance.sort_values('total_revenue', ascending=False)
        
        print(f"\n🏷️ Category Performance:")
        print(category_performance.to_string())
        
        # Customer segment insights
        segment_analysis = self.df.groupby('customer_segment').agg({
            'total_revenue': ['sum', 'mean', 'count'],
            'customer_id': 'nunique'
        }).round(2)
        
        segment_analysis.columns = ['total_revenue', 'avg_revenue', 'transactions', 'customers']
        print(f"\n👥 Customer Segment Analysis:")
        print(segment_analysis.to_string())
        
        # Store insights
        self.insights['revenue'] = {
            'growth_rate': monthly_growth,
            'top_category': category_performance.index[0],
            'best_segment': segment_analysis['total_revenue'].idxmax()
        }
        
        return monthly_revenue, category_performance
    
    def customer_intelligence(self):
        """Deep customer behavior analysis"""
        print("\n" + "="*50)
        print("👥 CUSTOMER INTELLIGENCE ANALYSIS")
        print("="*50)
        
        # Customer value analysis
        customer_metrics = self.df.groupby('customer_id').agg({
            'total_revenue': 'sum',
            'transaction_id': 'count',
            'date': ['min', 'max']
        })
        
        customer_metrics.columns = ['total_spent', 'purchases', 'first_date', 'last_date']
        customer_metrics['avg_order_value'] = customer_metrics['total_spent'] / customer_metrics['purchases']
        
        # Top customers
        top_customers = customer_metrics.nlargest(10, 'total_spent')
        print("🏆 Top 10 Customers by Value:")
        print(top_customers[['total_spent', 'purchases', 'avg_order_value']].to_string())
        
        # Pareto Analysis (80/20 rule)
        customer_sorted = customer_metrics.sort_values('total_spent', ascending=False)
        customer_sorted['cumulative_revenue'] = customer_sorted['total_spent'].cumsum()
        total_revenue = customer_sorted['total_spent'].sum()
        customer_sorted['revenue_percentage'] = customer_sorted['cumulative_revenue'] / total_revenue * 100
        
        # Find 80% revenue contributors
        customers_80_percent = len(customer_sorted[customer_sorted['revenue_percentage'] <= 80])
        pareto_percentage = customers_80_percent / len(customer_sorted) * 100
        
        print(f"\n📊 Pareto Analysis (80/20 Rule):")
        print(f"   • {customers_80_percent:,} customers ({pareto_percentage:.1f}%) generate 80% of revenue")
        print(f"   • Average high-value customer worth: ${customer_sorted.head(customers_80_percent)['total_spent'].mean():.2f}")
        
        # Purchase frequency analysis
        frequency = customer_metrics['purchases'].value_counts().sort_index()
        repeat_customers = frequency[frequency.index > 1].sum()
        repeat_rate = repeat_customers / len(customer_metrics) * 100
        
        print(f"\n🔄 Customer Loyalty Metrics:")
        print(f"   • One-time customers: {frequency.get(1, 0):,}")
        print(f"   • Repeat customers: {repeat_customers:,} ({repeat_rate:.1f}%)")
        
        return customer_metrics, pareto_percentage
    
    def create_executive_dashboard(self):
        """Generate professional business visualizations"""
        print("\n📊 Creating Executive Dashboard...")
        
        # Create comprehensive dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('E-Commerce Sales Analytics - Executive Dashboard', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # 1. Monthly Revenue Trend
        monthly_data = self.df.groupby('month_year')['total_revenue'].sum()
        monthly_data.plot(ax=axes[0,0], kind='line', marker='o', linewidth=2, color='#2E86AB')
        axes[0,0].set_title('Monthly Revenue Trend', fontweight='bold')
        axes[0,0].set_ylabel('Revenue ($)')
        axes[0,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # 2. Category Performance
        category_revenue = self.df.groupby('category')['total_revenue'].sum().sort_values()
        category_revenue.plot(ax=axes[0,1], kind='barh', color='#F18F01')
        axes[0,1].set_title('Revenue by Category', fontweight='bold')
        axes[0,1].set_xlabel('Revenue ($)')
        
        # 3. Customer Segment Distribution
        segment_counts = self.df['customer_segment'].value_counts()
        axes[0,2].pie(segment_counts.values, labels=segment_counts.index, 
                     autopct='%1.1f%%', startangle=90, colors=['#2E86AB', '#F18F01', '#C73E1D'])
        axes[0,2].set_title('Customer Segments', fontweight='bold')
        
        # 4. Weekly Pattern
        weekly_avg = self.df.groupby('day_of_week')['total_revenue'].mean()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_avg = weekly_avg.reindex(day_order)
        weekly_avg.plot(ax=axes[1,0], kind='bar', color='#A23B72')
        axes[1,0].set_title('Average Daily Revenue', fontweight='bold')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 5. Regional Performance
        regional_data = self.df.groupby('region')['total_revenue'].sum().sort_values()
        regional_data.plot(ax=axes[1,1], kind='barh', color='#C73E1D')
        axes[1,1].set_title('Revenue by Region', fontweight='bold')
        
        # 6. Price vs Quantity Analysis
        sample_data = self.df.sample(500)  # Sample for clarity
        scatter = axes[1,2].scatter(sample_data['unit_price'], sample_data['quantity'], 
                                   alpha=0.6, c=sample_data['total_revenue'], 
                                   cmap='viridis', s=30)
        axes[1,2].set_title('Price vs Quantity', fontweight='bold')
        axes[1,2].set_xlabel('Unit Price ($)')
        axes[1,2].set_ylabel('Quantity')
        
        plt.tight_layout()
        plt.savefig('outputs/charts/executive_dashboard.png', dpi=300, bbox_inches='tight')
        print("✅ Dashboard saved: outputs/charts/executive_dashboard.png")
        plt.show()
    
    def generate_business_insights(self):
        """Create actionable business recommendations"""
        print("\n" + "="*50)
        print("💡 STRATEGIC BUSINESS INSIGHTS")
        print("="*50)
        
        # Key findings
        total_revenue = self.df['total_revenue'].sum()
        top_category = self.df.groupby('category')['total_revenue'].sum().idxmax()
        best_segment = self.df.groupby('customer_segment')['total_revenue'].sum().idxmax()
        
        print(f"🎯 KEY PERFORMANCE INDICATORS:")
        print(f"   💰 Total Business Value: ${total_revenue:,.2f}")
        print(f"   🏆 Top Category: {top_category} (focus area for growth)")
        print(f"   👑 Best Segment: {best_segment} customers (highest value)")
        
        # Business recommendations
        recommendations = [
            f"Prioritize {top_category} inventory - drives highest revenue",
            f"Develop premium services for {best_segment} customers",
            "Implement customer loyalty program for repeat purchase incentives",
            "Optimize marketing spend during peak seasonal periods",
            "Focus retention campaigns on high-value customer segments"
        ]
        
        print(f"\n🚀 STRATEGIC RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        # Calculate business impact
        customer_value = self.df.groupby('customer_id')['total_revenue'].sum().mean()
        print(f"\n📊 BUSINESS IMPACT OPPORTUNITIES:")
        print(f"   • Customer retention improvement: +${customer_value * 0.15:.0f} per customer")
        print(f"   • Category optimization potential: +${total_revenue * 0.08:.0f} annually")
        print(f"   • Total identified opportunity: +${total_revenue * 0.12:.0f}")
        
        return recommendations
    
    def export_business_reports(self):
        """Export professional business reports"""
        print("\n💾 Exporting Business Intelligence Reports...")
        
        # Main dataset
        self.df.to_csv('data/ecommerce_transactions.csv', index=False)
        
        # Customer analysis
        customer_summary = self.df.groupby('customer_id').agg({
            'total_revenue': 'sum',
            'transaction_id': 'count',
            'category': lambda x: x.mode()[0]  # Most frequent category
        })
        customer_summary.columns = ['total_spent', 'transactions', 'preferred_category']
        customer_summary.to_csv('data/customer_analysis.csv')
        
        # Executive summary
        summary_data = {
            'Metric': ['Total Revenue', 'Transactions', 'Customers', 'Avg Order Value'],
            'Value': [
                f"${self.df['total_revenue'].sum():,.2f}",
                f"{len(self.df):,}",
                f"{self.df['customer_id'].nunique():,}",
                f"${self.df['total_revenue'].mean():.2f}"
            ]
        }
        pd.DataFrame(summary_data).to_csv('outputs/reports/executive_summary.csv', index=False)
        
        print("📄 Reports exported:")
        print("   • data/ecommerce_transactions.csv - Full dataset")
        print("   • data/customer_analysis.csv - Customer insights")
        print("   • outputs/reports/executive_summary.csv - Executive KPIs")

# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================
def main():
    """Execute complete data analytics workflow"""
    
    print("🎯 STARTING PROFESSIONAL DATA ANALYTICS PIPELINE")
    print("Demonstrating end-to-end business intelligence capabilities\n")
    
    # Step 1: Setup
    create_project_folders()
    
    # Step 2: Generate Business Data
    print("\n" + "="*60)
    print("STEP 1: DATA GENERATION")
    print("="*60)
    generator = BusinessDataGenerator(num_transactions=10000)
    df = generator.generate_ecommerce_data()
    
    # Step 3: Initialize Analytics
    print("\n" + "="*60)
    print("STEP 2: ANALYTICS INITIALIZATION") 
    print("="*60)
    analytics = ECommerceAnalytics(df)
    
    # Step 4: Business Analysis
    print("\n" + "="*60)
    print("STEP 3: COMPREHENSIVE BUSINESS ANALYSIS")
    print("="*60)
    
    analytics.business_overview()
    analytics.revenue_analysis()
    analytics.customer_intelligence()
    
    # Step 5: Executive Visualizations
    print("\n" + "="*60)
    print("STEP 4: EXECUTIVE DASHBOARD CREATION")
    print("="*60)
    analytics.create_executive_dashboard()
    
    # Step 6: Business Insights
    print("\n" + "="*60)
    print("STEP 5: STRATEGIC INSIGHTS GENERATION")
    print("="*60)
    analytics.generate_business_insights()
    
    # Step 7: Export Results
    print("\n" + "="*60)
    print("STEP 6: BUSINESS REPORT EXPORT")
    print("="*60)
    analytics.export_business_reports()
    
    # Final Summary
    print("\n" + "="*80)
    print("🎉 DATA ANALYTICS PROJECT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("📊 DELIVERABLES READY:")
    print("   ✅ Comprehensive business analysis")
    print("   ✅ Executive dashboard and visualizations")
    print("   ✅ Customer intelligence insights")
    print("   ✅ Strategic recommendations")
    print("   ✅ Exportable business reports")
    print("\n💼 This demonstrates professional data analyst capabilities:")
    print("   • Large dataset processing (10,000+ records)")
    print("   • Statistical analysis and business intelligence")
    print("   • Professional visualization and reporting")
    print("   • Actionable business insights generation")
    print("   • Stakeholder-ready deliverables")
    
    print(f"\n🚀 Ready for business impact and stakeholder presentation!")

# Execute the analytics pipeline
if __name__ == "__main__":
    main()
