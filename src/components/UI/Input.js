import React from 'react';
import styles from './Input.module.css';

const Input = React.forwardRef(
  (
    {
      label,
      type = 'text',
      value,
      onChange,
      onKeyDown,      // <-- important
      name,
      placeholder,
      error,
      required = false,
      disabled = false
    },
    ref
  ) => {
    return (
      <div className={styles.inputGroup}>
        {label && (
          <label htmlFor={name} className={styles.label}>
            {label}
            {required && "*"}
          </label>
        )}

        <input
          ref={ref}  // <-- this makes focus + keyDown work
          type={type}
          id={name}
          name={name}
          value={value}
          onChange={onChange}
          onKeyDown={onKeyDown}  // <-- now AIAssistant receives keyDown events
          placeholder={placeholder}
          className={`${styles.input} ${error ? styles.errorInput : ''}`}
          required={required}
          disabled={disabled}
        />

        {error && <p className={styles.errorMessage}>{error}</p>}
      </div>
    );
  }
);

export default Input;
