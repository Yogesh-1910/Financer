// src/components/Dashboard/BudgetExcelView.js
import React, { useState, useEffect, useMemo } from 'react';
import Card from '../UI/Card';
import Button from '../UI/Button';
import Input from '../UI/Input';
import styles from './BudgetExcelView.module.css';
import budgetService from '../../api/budgetService';
import * as XLSX from 'xlsx';
import { saveAs } from 'file-saver';
import { useAuth } from '../../contexts/AuthContext';

// --- Custom Event Names (Exported for DashboardHome to use) ---
export const USER_DATA_UPDATED_EVENT = 'userDataUpdated'; // For salary from localStorage
export const BUDGET_ITEMS_UPDATED_EVENT = 'budgetItemsUpdated'; // For backend budget items

const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', minimumFractionDigits: 0 }).format(amount || 0);
};
const getCurrentMonthYear = () => new Date().toISOString().slice(0, 7);

const BudgetExcelView = () => {
  const { currentUser } = useAuth();
  const [items, setItems] = useState([]); // For backend-driven items (expenses, other incomes)
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const initialNewItemState = {
    monthYear: getCurrentMonthYear(),
    category: 'Living Expenses',
    itemName: '',
    plannedAmount: '',
    type: 'expense',
    notes: ''
  };
  const [formVisible, setFormVisible] = useState(false);
  const [editingItem, setEditingItem] = useState(null);
  const [formData, setFormData] = useState(initialNewItemState);

  const salaryStorageKey = useMemo(() => `financeAppUserSalary_${currentUser?.id}`, [currentUser]);
  const [salaryInputValue, setSalaryInputValue] = useState('');

  // Load initial data: salary from localStorage, other items from backend
  useEffect(() => {
    const loadData = async () => {
      if (!currentUser?.id) {
        setIsLoading(false);
        setError("User not available. Please log in.");
        return;
      }
      setIsLoading(true);
      setError(null);
      try {
        const storedSalary = localStorage.getItem(salaryStorageKey);
        setSalaryInputValue(storedSalary || '0');

        const fetchedItems = await budgetService.getBudgetItems();
        setItems(fetchedItems || []);
        console.log("[BudgetExcelView] Initial data loaded. Salary (LS):", storedSalary, "Items (BE):", fetchedItems);
      } catch (err) {
        console.error("[BudgetExcelView] Failed to load initial data:", err);
        setError(err.response?.data?.msg || err.message || "Could not load budget data.");
      }
      setIsLoading(false);
    };
    loadData();
  }, [currentUser, salaryStorageKey]); // Rerun if user changes

  const handleSalaryInputChange = (e) => {
    setSalaryInputValue(e.target.value);
  };

  const handleSaveSalary = async () => {
    if (!currentUser?.id) return;
    const amount = parseFloat(salaryInputValue);
    if (isNaN(amount) || amount < 0) {
      setError("Please enter a valid positive salary amount.");
      setTimeout(() => setError(null), 3000);
      return;
    }
    setError(null);
    try {
      localStorage.setItem(salaryStorageKey, amount.toString());
      console.log("[BudgetExcelView] Salary saved to localStorage:", amount);
      window.dispatchEvent(new CustomEvent(USER_DATA_UPDATED_EVENT, { detail: { salary: amount } }));
      alert("Salary updated successfully!");
    } catch (err) {
      console.error("[BudgetExcelView] Failed to save salary:", err);
      setError("Could not save salary.");
    }
  };

  const handleFormInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const openFormForAdd = () => {
    setEditingItem(null);
    setFormData(initialNewItemState);
    setFormVisible(true);
    setError(null);
  };

  const openFormForEdit = (item) => {
    setEditingItem(item);
    setFormData({
      ...item,
      plannedAmount: item.plannedAmount !== null ? String(item.plannedAmount) : '',
      actualAmount: item.actualAmount !== null ? String(item.actualAmount) : '',
      notes: item.notes || ''
    });
    setFormVisible(true);
    setError(null);
  };

  const closeForm = () => {
    setFormVisible(false);
    setEditingItem(null);
    setFormData(initialNewItemState);
    setError(null);
  };

  const handleSubmitItem = async (e) => {
    e.preventDefault();
    setError(null);
    const itemDataToSubmit = editingItem || formData;
    const plannedAmt = parseFloat(itemDataToSubmit.plannedAmount);

    if (!itemDataToSubmit.itemName || !itemDataToSubmit.monthYear || !itemDataToSubmit.category || !itemDataToSubmit.type || isNaN(plannedAmt)) {
      setError("Month/Year, Item Name, Category, Type, and a valid Planned Amount are required.");
      return;
    }
    // Ensure type is not 'income' AND category is 'Salary' if salary is managed separately
    if (itemDataToSubmit.type === 'income' && itemDataToSubmit.category?.toLowerCase() === 'salary') {
        setError("Please manage 'Monthly Salary' through the dedicated salary input field above.");
        return;
    }

    const payload = {
      ...itemDataToSubmit,
      plannedAmount: plannedAmt,
      actualAmount: itemDataToSubmit.actualAmount ? parseFloat(itemDataToSubmit.actualAmount) : null,
      notes: itemDataToSubmit.notes || null,
    };
    if (!editingItem && payload.id) delete payload.id; // Remove ID for new items

    console.log("[BudgetExcelView] Submitting budget item to backend:", payload);
    try {
      let result;
      if (editingItem) {
        result = await budgetService.updateBudgetItem(editingItem.id, payload);
        setItems(prev => prev.map(it => it.id === result.id ? result : it));
      } else {
        result = await budgetService.addBudgetItem(payload);
        setItems(prev => [...prev, result]);
      }
      closeForm();
      window.dispatchEvent(new CustomEvent(BUDGET_ITEMS_UPDATED_EVENT));
      console.log("[BudgetExcelView] Budget item saved successfully.");
    } catch (err) {
      console.error("[BudgetExcelView] Failed to save budget item:", err.response?.data || err.message);
      setError(err.response?.data?.msg || err.response?.data?.error || "Could not save budget item.");
    }
  };

  const handleRemoveItem = async (id) => {
    if (window.confirm("Are you sure you want to delete this item?")) {
      setError(null);
      try {
        await budgetService.deleteBudgetItem(id);
        setItems(prev => prev.filter(item => item.id !== id));
        window.dispatchEvent(new CustomEvent(BUDGET_ITEMS_UPDATED_EVENT));
        console.log("[BudgetExcelView] Budget item deleted successfully.");
      } catch (err) {
        console.error("[BudgetExcelView] Failed to delete item:", err.response?.data || err.message);
        setError(err.response?.data?.msg || err.response?.data?.error || "Could not delete item.");
      }
    }
  };

  // --- Memoized Calculations ---
  const currentSalaryFromStorage = useMemo(() => {
    if (!currentUser?.id) return 0;
    return parseFloat(localStorage.getItem(salaryStorageKey)) || 0;
  }, [salaryStorageKey, items]); // Re-calculate if items change to force re-render from event

  const otherIncomeFromItems = useMemo(() => {
    return items
      .filter(item => item.type === 'income' && item.category?.toLowerCase() !== 'salary')
      .reduce((sum, item) => sum + (parseFloat(item.plannedAmount) || 0), 0);
  }, [items]);

  const totalIncomeForDisplay = useMemo(() => currentSalaryFromStorage + otherIncomeFromItems, [currentSalaryFromStorage, otherIncomeFromItems]);

  const totalExpenses = useMemo(() => {
    return items
      .filter(item => item.type === 'expense')
      .reduce((sum, item) => sum + (parseFloat(item.plannedAmount) || 0), 0);
  }, [items]);

  const netSavings = useMemo(() => totalIncomeForDisplay - totalExpenses, [totalIncomeForDisplay, totalExpenses]);

  const budgetSummaryForTable = useMemo(() => {
    const summary = [];
    // Add salary row from localStorage
    summary.push({
      id: 'salary-ls', // Unique key for table
      monthYear: getCurrentMonthYear(), // Or a more general label
      category: 'Salary',
      itemName: 'Monthly Salary (from settings)',
      type: 'income',
      monthly: currentSalaryFromStorage,
      total: currentSalaryFromStorage * 12,
      isLocalStorageSalary: true // Flag to prevent edit/delete buttons for this row
    });

    // Add other income and expense items from backend
    items.forEach(it => {
      summary.push({ ...it, monthly: it.plannedAmount, total: (it.plannedAmount || 0) * 12 });
    });

    // Add summary totals
    summary.push({ category: 'Summary', itemName: 'Total Income', monthly: totalIncomeForDisplay, total: totalIncomeForDisplay * 12, isHeader: true, isIncomeTotal: true, type: 'summary' });
    summary.push({ category: 'Summary', itemName: 'Total Expenses', monthly: totalExpenses, total: totalExpenses * 12, isHeader: true, isExpenseTotal: true, type: 'summary' });
    summary.push({ category: 'Summary', itemName: 'Net Savings', monthly: netSavings, total: netSavings * 12, isHeader: true, isSavings: true, type: 'summary' });
    return summary;
  }, [items, currentSalaryFromStorage, totalIncomeForDisplay, totalExpenses, netSavings]);

  const handleExportToExcel = () => {
    console.log("[BudgetExcelView] Exporting to Excel...");
    const dataForExport = budgetSummaryForTable
        .map(row => ({ // No filter, export all including salary from LS
            Month: row.monthYear,
            Category: row.category,
            Item: row.itemName,
            Type: row.type,
            'Planned Monthly (INR)': row.monthly,
            'Est. Yearly / Total (INR)': row.total
        }));
    if (dataForExport.length === 0) {
        alert("No data available to export."); return;
    }
    const ws = XLSX.utils.json_to_sheet(dataForExport);
    const columnWidths = [ { wch: 15 }, { wch: 20 }, { wch: 30 }, {wch: 10}, { wch: 20 }, { wch: 25 } ];
    ws['!cols'] = columnWidths;
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, 'Budget Overview');
    const excelBuffer = XLSX.write(wb, { bookType: 'xlsx', type: 'array' });
    const dataBlob = new Blob([excelBuffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;charset=UTF-8' });
    const fileName = `budget_overview_${getCurrentMonthYear()}.xlsx`;
    saveAs(dataBlob, fileName);
  };

  if (isLoading && !currentUser?.id) { // Show initial loading or user not found
     return <Card title="Budget Manager"><p>Loading user data or user not logged in...</p></Card>;
  }
  if (isLoading) { // Loading budget items
     return <Card title="Budget Manager"><p className={styles.loadingMessage}>Loading budget data...</p></Card>;
  }


  return (
    <Card title="Budget Manager" actions={!formVisible && <Button onClick={openFormForAdd}>+ Add Budget Item</Button>}>
      {error && <p className={styles.errorMessage}>{error}</p>}
      <div className={styles.controls}>
        <label htmlFor="salaryFigure" className={styles.salaryLabel}>Your Gross Monthly Salary (INR):</label>
        <div style={{display: 'flex', gap: '10px', alignItems: 'center'}}>
            <Input
            id="salaryFigure"
            type="number"
            value={salaryInputValue}
            onChange={handleSalaryInputChange}
            name="salaryFigure"
            placeholder="Enter salary"
            className={styles.salaryInput}
            />
            <Button onClick={handleSaveSalary} variant="secondary" className={styles.saveSalaryButton}>Save Salary</Button>
        </div>
      </div>

      {formVisible && (
           <form onSubmit={handleSubmitItem} className={styles.addItemForm}>
             <h4>{editingItem ? 'Edit Budget Item' : 'Add New Budget Item'}</h4>
             <div className={styles.formRow}>
                <Input label="Month (YYYY-MM)" name="monthYear" type="month" value={formData.monthYear} onChange={handleFormInputChange} required />
                <Input label="Item Name" name="itemName" placeholder="e.g., Rent, Mutual Fund" value={formData.itemName} onChange={handleFormInputChange} required />
             </div>
             <div className={styles.formRow}>
                <Input label="Planned Amount" name="plannedAmount" type="number" step="0.01" placeholder="Monthly Amount" value={String(formData.plannedAmount)} onChange={handleFormInputChange} required />
                <Input label="Actual Amount (Optional)" name="actualAmount" type="number" step="0.01" placeholder="Actual Spent/Received" value={String(formData.actualAmount)} onChange={handleFormInputChange} />
             </div>
             <div className={styles.formRow}>
            <div className={styles.inputGroup}>
              <label htmlFor="category">Category</label>
              <select
                id="category"
                name="category"
                value={formData.category}
                onChange={handleFormInputChange}
                className={styles.selectInput}
                required
              >
                <option value="">-- Select Category --</option>
                <option value="Other Income">Other Income</option>
                <option value="Housing">Housing (Rent/Mortgage)</option>
                <option value="Utilities">Utilities</option>
                <option value="Transportation">Transportation</option>
                <option value="Groceries">Groceries</option>
                <option value="EMI">EMI / Loan Payment</option>
                <option value="Investment">Investment/Savings</option>
                <option value="Insurance">Insurance</option>
                <option value="Healthcare">Healthcare</option>
                <option value="Entertainment">Entertainment</option>
                <option value="Education">Education</option>
                <option value="Other Expense">Other Expense</option>
              </select>
            </div>

            <div className={styles.inputGroup}>
              <label htmlFor="type">Type</label>
              <select
                id="type"
                name="type"
                value={formData.type}
                onChange={handleFormInputChange}
                className={styles.selectInput}
                required
              >
                <option value="expense">Expense</option>
                <option value="income">Income</option>
              </select>
            </div>
          </div>

             <Input label="Notes (Optional)" name="notes" value={formData.notes || ''} onChange={handleFormInputChange} />
             <div className={styles.formActions}>
               <Button type="submit">{editingItem ? 'Save Changes' : 'Add Item'}</Button>
               <Button type="button" variant="secondary" onClick={closeForm}>Cancel</Button>
             </div>
           </form>
         )}

        <div className={styles.tableContainer}>
        <table className={styles.budgetTable}>
            <thead>
            <tr><th>Month</th><th>Category</th><th>Item Name</th><th>Type</th><th>Planned Amount</th><th>Actions</th></tr>
            </thead>
            <tbody>
            {budgetSummaryForTable.map((item, index) => (
                <tr key={item.id || `summary-${index}-${item.itemName}`}
                    className={` ${item.isHeader && styles.headerRow} ${item.type === 'income' && !item.isHeader && styles.incomeRowActual} ${item.isPlaceholder && styles.placeholderRow} ${item.isSavings && styles.savingsRow} ${item.isExpenseTotal && styles.expenseTotalRow} ${item.isIncomeTotal && styles.incomeTotalRow} `}>
                <td>{item.isHeader ? <b>{item.category}</b> : item.monthYear}</td>
                <td>{item.isHeader ? '' : item.category}</td>
                <td>{item.isHeader ? <b>{item.itemName}</b> : item.itemName}</td>
                <td>{item.isHeader ? '' : item.type}</td>
                <td className={styles.amountCell}>{formatCurrency(item.monthly)}</td>
                <td>
                    {!item.isHeader && !item.isLocalStorageSalary && item.id && ( // Don't show edit/delete for localStorage salary row
                    <>
                        <Button onClick={() => openFormForEdit(item)} variant="text" className={styles.actionButtonSmall} title="Edit">✎</Button>
                        <Button onClick={() => handleRemoveItem(item.id)} variant="text" className={`${styles.actionButtonSmall} ${styles.deleteButton}`} title="Delete">✕</Button>
                    </>
                    )}
                </td></tr>))}
            </tbody></table>
        </div>
        {(items.length > 0 || currentSalaryFromStorage > 0) && <Button onClick={handleExportToExcel} className={styles.exportButton}>Export to Excel</Button>}
    </Card>
  );
};

export default BudgetExcelView;