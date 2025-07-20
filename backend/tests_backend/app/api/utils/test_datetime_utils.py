"""
🎵 Spotify AI Agent - Tests DateTime Utils Module
=================================================

Tests enterprise complets pour le module datetime_utils
avec validation de temporalité, fuseaux horaires et performance.

🎖️ Développé par l'équipe d'experts enterprise
"""

import pytest
from datetime import datetime, date, time, timedelta, timezone
from dateutil.tz import tzutc, tzlocal, gettz
from unittest.mock import patch, Mock
import pytz

# Import du module à tester
from backend.app.api.utils.datetime_utils import (
    now,
    utc_now,
    parse_datetime,
    format_datetime,
    convert_timezone,
    add_time,
    subtract_time,
    time_difference,
    is_business_day,
    get_business_days,
    start_of_day,
    end_of_day,
    start_of_week,
    end_of_week,
    start_of_month,
    end_of_month,
    age_from_birthdate,
    is_weekend,
    get_quarter,
    days_until,
    format_relative_time,
    timestamp_to_datetime,
    datetime_to_timestamp,
    validate_date_range,
    get_timezone_offset
)

from . import TestUtils, security_test, performance_test, integration_test


class TestDateTimeUtils:
    """Tests pour le module datetime_utils"""
    
    def test_now_basic(self):
        """Test obtention heure actuelle"""
        current = now()
        
        assert isinstance(current, datetime)
        assert current.tzinfo is not None  # Doit avoir une timezone
    
    def test_now_timezone(self):
        """Test heure actuelle avec timezone"""
        paris_tz = pytz.timezone('Europe/Paris')
        current_paris = now(timezone=paris_tz)
        
        assert current_paris.tzinfo == paris_tz
    
    def test_utc_now_basic(self):
        """Test heure UTC actuelle"""
        utc_current = utc_now()
        
        assert isinstance(utc_current, datetime)
        assert utc_current.tzinfo == pytz.UTC
    
    def test_parse_datetime_iso_format(self):
        """Test parsing format ISO"""
        iso_string = "2025-07-14T10:30:00Z"
        parsed = parse_datetime(iso_string)
        
        assert isinstance(parsed, datetime)
        assert parsed.year == 2025
        assert parsed.month == 7
        assert parsed.day == 14
        assert parsed.hour == 10
        assert parsed.minute == 30
    
    def test_parse_datetime_custom_format(self):
        """Test parsing format personnalisé"""
        date_string = "14/07/2025 10:30"
        format_string = "%d/%m/%Y %H:%M"
        parsed = parse_datetime(date_string, format_string)
        
        assert parsed.year == 2025
        assert parsed.month == 7
        assert parsed.day == 14
        assert parsed.hour == 10
        assert parsed.minute == 30
    
    def test_parse_datetime_multiple_formats(self):
        """Test parsing avec formats multiples"""
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%Y-%m-%dT%H:%M:%SZ"
        ]
        
        # Test différents formats
        dates = [
            "2025-07-14 10:30:00",
            "14/07/2025 10:30",
            "2025-07-14T10:30:00Z"
        ]
        
        for date_string in dates:
            parsed = parse_datetime(date_string, formats)
            assert isinstance(parsed, datetime)
    
    def test_parse_datetime_invalid(self):
        """Test parsing date invalide"""
        invalid_date = "not-a-date"
        result = parse_datetime(invalid_date)
        
        assert result is None  # Ou exception selon implémentation
    
    def test_format_datetime_basic(self):
        """Test formatage datetime basique"""
        dt = datetime(2025, 7, 14, 10, 30, 0)
        formatted = format_datetime(dt)
        
        assert isinstance(formatted, str)
        assert "2025" in formatted
        assert "07" in formatted or "7" in formatted
        assert "14" in formatted
    
    def test_format_datetime_custom_format(self):
        """Test formatage format personnalisé"""
        dt = datetime(2025, 7, 14, 10, 30, 0)
        formatted = format_datetime(dt, format_string="%d/%m/%Y %H:%M")
        
        assert formatted == "14/07/2025 10:30"
    
    def test_format_datetime_locale(self):
        """Test formatage avec locale"""
        dt = datetime(2025, 7, 14, 10, 30, 0)
        
        # Format français
        formatted_fr = format_datetime(dt, locale='fr_FR', 
                                     format_string="%A %d %B %Y")
        
        # Doit contenir des mots français (si locale supportée)
        if formatted_fr != format_datetime(dt, format_string="%A %d %B %Y"):
            assert any(word in formatted_fr.lower() for word in 
                      ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche'])
    
    def test_convert_timezone_basic(self):
        """Test conversion timezone basique"""
        # UTC vers Paris
        utc_dt = datetime(2025, 7, 14, 10, 30, 0, tzinfo=pytz.UTC)
        paris_tz = pytz.timezone('Europe/Paris')
        
        paris_dt = convert_timezone(utc_dt, paris_tz)
        
        assert paris_dt.tzinfo == paris_tz
        assert paris_dt.hour != utc_dt.hour  # Décalage horaire
    
    def test_convert_timezone_string(self):
        """Test conversion timezone avec string"""
        utc_dt = datetime(2025, 7, 14, 10, 30, 0, tzinfo=pytz.UTC)
        
        ny_dt = convert_timezone(utc_dt, 'America/New_York')
        
        assert ny_dt.tzinfo.zone == 'America/New_York'
    
    def test_add_time_basic(self):
        """Test ajout de temps"""
        dt = datetime(2025, 7, 14, 10, 30, 0)
        
        # Ajouter 2 heures
        new_dt = add_time(dt, hours=2)
        assert new_dt.hour == 12
        
        # Ajouter 30 minutes
        new_dt = add_time(dt, minutes=30)
        assert new_dt.minute == 0  # 10:30 + 30min = 11:00
        assert new_dt.hour == 11
        
        # Ajouter 1 jour
        new_dt = add_time(dt, days=1)
        assert new_dt.day == 15
    
    def test_add_time_complex(self):
        """Test ajout de temps complexe"""
        dt = datetime(2025, 7, 14, 23, 45, 0)
        
        # Ajouter 30 minutes (passe au jour suivant)
        new_dt = add_time(dt, minutes=30)
        assert new_dt.day == 15
        assert new_dt.hour == 0
        assert new_dt.minute == 15
    
    def test_subtract_time_basic(self):
        """Test soustraction de temps"""
        dt = datetime(2025, 7, 14, 10, 30, 0)
        
        # Soustraire 2 heures
        new_dt = subtract_time(dt, hours=2)
        assert new_dt.hour == 8
        
        # Soustraire 1 jour
        new_dt = subtract_time(dt, days=1)
        assert new_dt.day == 13
    
    def test_time_difference_basic(self):
        """Test différence de temps"""
        dt1 = datetime(2025, 7, 14, 10, 0, 0)
        dt2 = datetime(2025, 7, 14, 12, 30, 0)
        
        diff = time_difference(dt2, dt1)
        
        assert isinstance(diff, timedelta)
        assert diff.total_seconds() == 2.5 * 3600  # 2h30 en secondes
    
    def test_time_difference_units(self):
        """Test différence de temps en unités"""
        dt1 = datetime(2025, 7, 14, 10, 0, 0)
        dt2 = datetime(2025, 7, 15, 12, 30, 0)  # +1 jour 2h30
        
        # En heures
        hours_diff = time_difference(dt2, dt1, unit='hours')
        assert hours_diff == 26.5  # 24h + 2h30
        
        # En jours
        days_diff = time_difference(dt2, dt1, unit='days')
        assert abs(days_diff - 1.104) < 0.01  # ~1.1 jours
    
    def test_is_business_day_weekday(self):
        """Test jour ouvrable - jour de semaine"""
        # Lundi 14 juillet 2025
        monday = datetime(2025, 7, 14)
        assert is_business_day(monday) is True
        
        # Mardi
        tuesday = datetime(2025, 7, 15)
        assert is_business_day(tuesday) is True
        
        # Vendredi
        friday = datetime(2025, 7, 18)
        assert is_business_day(friday) is True
    
    def test_is_business_day_weekend(self):
        """Test jour ouvrable - weekend"""
        # Samedi 19 juillet 2025
        saturday = datetime(2025, 7, 19)
        assert is_business_day(saturday) is False
        
        # Dimanche
        sunday = datetime(2025, 7, 20)
        assert is_business_day(sunday) is False
    
    def test_is_business_day_holidays(self):
        """Test jour ouvrable avec jours fériés"""
        # 14 juillet 2025 (Fête nationale française)
        bastille_day = datetime(2025, 7, 14)
        
        france_holidays = ['2025-07-14']  # Liste des jours fériés
        result = is_business_day(bastille_day, holidays=france_holidays)
        
        assert result is False
    
    def test_get_business_days_basic(self):
        """Test obtention jours ouvrables"""
        start_date = datetime(2025, 7, 14)  # Lundi
        end_date = datetime(2025, 7, 18)    # Vendredi
        
        business_days = get_business_days(start_date, end_date)
        
        assert len(business_days) == 5  # Lundi à vendredi
        assert all(is_business_day(day) for day in business_days)
    
    def test_get_business_days_with_weekend(self):
        """Test jours ouvrables incluant weekend"""
        start_date = datetime(2025, 7, 14)  # Lundi
        end_date = datetime(2025, 7, 20)    # Dimanche
        
        business_days = get_business_days(start_date, end_date)
        
        assert len(business_days) == 5  # Seulement les jours ouvrables
        assert not any(day.weekday() >= 5 for day in business_days)
    
    def test_start_of_day(self):
        """Test début de journée"""
        dt = datetime(2025, 7, 14, 15, 30, 45, 123456)
        start = start_of_day(dt)
        
        assert start.year == 2025
        assert start.month == 7
        assert start.day == 14
        assert start.hour == 0
        assert start.minute == 0
        assert start.second == 0
        assert start.microsecond == 0
    
    def test_end_of_day(self):
        """Test fin de journée"""
        dt = datetime(2025, 7, 14, 15, 30, 45)
        end = end_of_day(dt)
        
        assert end.year == 2025
        assert end.month == 7
        assert end.day == 14
        assert end.hour == 23
        assert end.minute == 59
        assert end.second == 59
        assert end.microsecond == 999999
    
    def test_start_of_week(self):
        """Test début de semaine"""
        # Mercredi 16 juillet 2025
        dt = datetime(2025, 7, 16)
        start = start_of_week(dt)
        
        # Doit être lundi 14 juillet
        assert start.weekday() == 0  # Lundi
        assert start.day == 14
        assert start.hour == 0
        assert start.minute == 0
    
    def test_start_of_week_sunday_first(self):
        """Test début de semaine - dimanche premier"""
        dt = datetime(2025, 7, 16)  # Mercredi
        start = start_of_week(dt, week_start='sunday')
        
        # Doit être dimanche 13 juillet
        assert start.weekday() == 6  # Dimanche
        assert start.day == 13
    
    def test_end_of_week(self):
        """Test fin de semaine"""
        dt = datetime(2025, 7, 16)  # Mercredi
        end = end_of_week(dt)
        
        # Doit être dimanche 20 juillet
        assert end.weekday() == 6  # Dimanche
        assert end.day == 20
        assert end.hour == 23
        assert end.minute == 59
    
    def test_start_of_month(self):
        """Test début de mois"""
        dt = datetime(2025, 7, 16, 15, 30, 45)
        start = start_of_month(dt)
        
        assert start.year == 2025
        assert start.month == 7
        assert start.day == 1
        assert start.hour == 0
        assert start.minute == 0
    
    def test_end_of_month(self):
        """Test fin de mois"""
        dt = datetime(2025, 7, 16)
        end = end_of_month(dt)
        
        assert end.year == 2025
        assert end.month == 7
        assert end.day == 31  # Juillet a 31 jours
        assert end.hour == 23
        assert end.minute == 59
    
    def test_end_of_month_february(self):
        """Test fin de mois - février"""
        dt = datetime(2025, 2, 15)  # Février 2025 (pas bissextile)
        end = end_of_month(dt)
        
        assert end.day == 28
        
        # Test année bissextile
        dt_leap = datetime(2024, 2, 15)  # 2024 est bissextile
        end_leap = end_of_month(dt_leap)
        
        assert end_leap.day == 29
    
    def test_age_from_birthdate_basic(self):
        """Test calcul âge basique"""
        # Né le 14 juillet 1990
        birthdate = datetime(1990, 7, 14)
        reference_date = datetime(2025, 7, 14)  # 35ème anniversaire
        
        age = age_from_birthdate(birthdate, reference_date)
        
        assert age == 35
    
    def test_age_from_birthdate_before_birthday(self):
        """Test calcul âge avant anniversaire"""
        birthdate = datetime(1990, 7, 14)
        reference_date = datetime(2025, 7, 13)  # Veille de l'anniversaire
        
        age = age_from_birthdate(birthdate, reference_date)
        
        assert age == 34
    
    def test_age_from_birthdate_current(self):
        """Test calcul âge actuel"""
        birthdate = datetime(1990, 7, 14)
        age = age_from_birthdate(birthdate)  # Date actuelle
        
        assert isinstance(age, int)
        assert age >= 0
    
    def test_is_weekend_basic(self):
        """Test détection weekend"""
        # Samedi
        saturday = datetime(2025, 7, 19)
        assert is_weekend(saturday) is True
        
        # Dimanche
        sunday = datetime(2025, 7, 20)
        assert is_weekend(sunday) is True
        
        # Lundi
        monday = datetime(2025, 7, 21)
        assert is_weekend(monday) is False
    
    def test_get_quarter_basic(self):
        """Test obtention trimestre"""
        assert get_quarter(datetime(2025, 1, 15)) == 1  # Janvier = Q1
        assert get_quarter(datetime(2025, 4, 15)) == 2  # Avril = Q2
        assert get_quarter(datetime(2025, 7, 15)) == 3  # Juillet = Q3
        assert get_quarter(datetime(2025, 10, 15)) == 4  # Octobre = Q4
    
    def test_get_quarter_edge_cases(self):
        """Test trimestre cas limites"""
        assert get_quarter(datetime(2025, 3, 31)) == 1  # Fin Q1
        assert get_quarter(datetime(2025, 4, 1)) == 2   # Début Q2
        assert get_quarter(datetime(2025, 12, 31)) == 4 # Fin Q4
    
    def test_days_until_basic(self):
        """Test jours jusqu'à date"""
        start_date = datetime(2025, 7, 14)
        target_date = datetime(2025, 7, 17)
        
        days = days_until(target_date, start_date)
        
        assert days == 3
    
    def test_days_until_past(self):
        """Test jours jusqu'à date passée"""
        start_date = datetime(2025, 7, 14)
        past_date = datetime(2025, 7, 10)
        
        days = days_until(past_date, start_date)
        
        assert days == -4  # Négatif pour le passé
    
    def test_format_relative_time_recent(self):
        """Test formatage temps relatif récent"""
        now_dt = datetime(2025, 7, 14, 10, 30, 0)
        
        # Il y a 5 minutes
        past_dt = now_dt - timedelta(minutes=5)
        relative = format_relative_time(past_dt, now_dt)
        
        assert "5 minute" in relative.lower() or "minute" in relative.lower()
        assert "ago" in relative.lower() or "il y a" in relative.lower()
    
    def test_format_relative_time_hours(self):
        """Test formatage temps relatif en heures"""
        now_dt = datetime(2025, 7, 14, 10, 30, 0)
        
        # Il y a 3 heures
        past_dt = now_dt - timedelta(hours=3)
        relative = format_relative_time(past_dt, now_dt)
        
        assert "3" in relative and ("hour" in relative.lower() or "heure" in relative.lower())
    
    def test_format_relative_time_future(self):
        """Test formatage temps relatif futur"""
        now_dt = datetime(2025, 7, 14, 10, 30, 0)
        
        # Dans 2 heures
        future_dt = now_dt + timedelta(hours=2)
        relative = format_relative_time(future_dt, now_dt)
        
        assert "2" in relative
        assert "in" in relative.lower() or "dans" in relative.lower()
    
    def test_timestamp_to_datetime_basic(self):
        """Test conversion timestamp vers datetime"""
        timestamp = 1721824200  # Timestamp Unix
        dt = timestamp_to_datetime(timestamp)
        
        assert isinstance(dt, datetime)
        assert dt.year >= 2024  # Vérifie cohérence
    
    def test_timestamp_to_datetime_milliseconds(self):
        """Test conversion timestamp millisecondes"""
        timestamp_ms = 1721824200000  # Timestamp en millisecondes
        dt = timestamp_to_datetime(timestamp_ms, unit='ms')
        
        assert isinstance(dt, datetime)
    
    def test_datetime_to_timestamp_basic(self):
        """Test conversion datetime vers timestamp"""
        dt = datetime(2025, 7, 14, 10, 30, 0, tzinfo=pytz.UTC)
        timestamp = datetime_to_timestamp(dt)
        
        assert isinstance(timestamp, (int, float))
        assert timestamp > 1700000000  # Vérifie cohérence (après 2023)
    
    def test_timestamp_datetime_roundtrip(self):
        """Test aller-retour timestamp/datetime"""
        original_dt = datetime(2025, 7, 14, 10, 30, 0, tzinfo=pytz.UTC)
        
        timestamp = datetime_to_timestamp(original_dt)
        converted_dt = timestamp_to_datetime(timestamp)
        
        # Comparaison avec tolérance (précision)
        diff = abs((original_dt - converted_dt).total_seconds())
        assert diff < 1  # Moins d'une seconde de différence
    
    def test_validate_date_range_valid(self):
        """Test validation plage de dates valide"""
        start_date = datetime(2025, 7, 14)
        end_date = datetime(2025, 7, 20)
        
        result = validate_date_range(start_date, end_date)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
    
    def test_validate_date_range_invalid(self):
        """Test validation plage de dates invalide"""
        start_date = datetime(2025, 7, 20)
        end_date = datetime(2025, 7, 14)  # Fin avant début
        
        result = validate_date_range(start_date, end_date)
        
        assert result['valid'] is False
        assert len(result['errors']) > 0
    
    def test_validate_date_range_constraints(self):
        """Test validation avec contraintes"""
        start_date = datetime(2025, 7, 14)
        end_date = datetime(2025, 8, 14)  # 1 mois
        
        # Contrainte: maximum 20 jours
        result = validate_date_range(start_date, end_date, max_days=20)
        
        assert result['valid'] is False
        
        # Contrainte: minimum 5 jours
        short_end = datetime(2025, 7, 16)  # 2 jours
        result = validate_date_range(start_date, short_end, min_days=5)
        
        assert result['valid'] is False
    
    def test_get_timezone_offset_basic(self):
        """Test obtention décalage timezone"""
        utc_dt = datetime(2025, 7, 14, 12, 0, 0, tzinfo=pytz.UTC)
        
        # Paris en été (UTC+2)
        paris_tz = pytz.timezone('Europe/Paris')
        offset = get_timezone_offset(utc_dt, paris_tz)
        
        assert offset == 2.0  # +2 heures en été
    
    def test_get_timezone_offset_winter(self):
        """Test décalage timezone en hiver"""
        winter_dt = datetime(2025, 1, 14, 12, 0, 0, tzinfo=pytz.UTC)
        
        # Paris en hiver (UTC+1)
        paris_tz = pytz.timezone('Europe/Paris')
        offset = get_timezone_offset(winter_dt, paris_tz)
        
        assert offset == 1.0  # +1 heure en hiver
    
    @performance_test
    def test_datetime_operations_performance(self):
        """Test performance opérations datetime"""
        def datetime_operations():
            current = now()
            utc_current = utc_now()
            
            # 1000 opérations de manipulation
            for i in range(1000):
                added = add_time(current, minutes=i)
                subtracted = subtract_time(added, hours=1)
                start_day = start_of_day(subtracted)
                end_day = end_of_day(start_day)
            
            return True
        
        TestUtils.assert_performance(datetime_operations, max_time_ms=500)
    
    @performance_test
    def test_timezone_conversion_performance(self):
        """Test performance conversions timezone"""
        dt = datetime(2025, 7, 14, 10, 30, 0, tzinfo=pytz.UTC)
        timezones = [
            'Europe/Paris',
            'America/New_York',
            'Asia/Tokyo',
            'Australia/Sydney'
        ]
        
        def convert_many_timezones():
            for _ in range(100):
                for tz_name in timezones:
                    converted = convert_timezone(dt, tz_name)
            return True
        
        TestUtils.assert_performance(convert_many_timezones, max_time_ms=200)
    
    @integration_test
    def test_complete_datetime_workflow(self):
        """Test workflow complet datetime"""
        # Scénario: Planning de tâches avec fuseaux horaires
        
        # 1. Créer événement
        event_start = parse_datetime("2025-07-14T09:00:00Z")
        assert event_start is not None
        
        # 2. Convertir en timezone locale
        paris_tz = pytz.timezone('Europe/Paris')
        local_start = convert_timezone(event_start, paris_tz)
        
        # 3. Calculer fin d'événement (2 heures)
        local_end = add_time(local_start, hours=2)
        
        # 4. Vérifier si jour ouvrable
        is_workday = is_business_day(local_start)
        
        # 5. Calculer durée
        duration = time_difference(local_end, local_start, unit='hours')
        
        # 6. Obtenir informations temporelles
        quarter = get_quarter(local_start)
        is_wknd = is_weekend(local_start)
        
        # 7. Formater pour affichage
        formatted = format_datetime(local_start, "%d/%m/%Y %H:%M")
        relative = format_relative_time(local_start)
        
        # 8. Convertir en timestamp pour stockage
        timestamp = datetime_to_timestamp(local_start)
        
        # Vérifications
        assert isinstance(local_start, datetime)
        assert local_start.tzinfo == paris_tz
        assert duration == 2.0
        assert quarter in [1, 2, 3, 4]
        assert isinstance(is_workday, bool)
        assert isinstance(is_wknd, bool)
        assert isinstance(formatted, str)
        assert isinstance(relative, str)
        assert isinstance(timestamp, (int, float))


# Tests de cas limites
class TestDateTimeUtilsEdgeCases:
    """Tests pour les cas limites et gestion d'erreurs"""
    
    def test_leap_year_handling(self):
        """Test gestion années bissextiles"""
        # 29 février 2024 (bissextile)
        leap_date = datetime(2024, 2, 29)
        
        # Ajouter 1 an (2025 n'est pas bissextile)
        next_year = add_time(leap_date, years=1)
        
        # Doit gérer gracieusement (28 février ou 1er mars)
        assert next_year.year == 2025
        assert next_year.month in [2, 3]
    
    def test_daylight_saving_transitions(self):
        """Test transitions heure d'été/hiver"""
        paris_tz = pytz.timezone('Europe/Paris')
        
        # Avant transition heure d'été (mars)
        before_dst = datetime(2025, 3, 29, 2, 30, 0)
        before_localized = paris_tz.localize(before_dst, is_dst=False)
        
        # Après transition
        after_dst = datetime(2025, 3, 30, 3, 30, 0)
        after_localized = paris_tz.localize(after_dst, is_dst=True)
        
        # Vérifier décalages
        offset_before = get_timezone_offset(before_localized.astimezone(pytz.UTC), paris_tz)
        offset_after = get_timezone_offset(after_localized.astimezone(pytz.UTC), paris_tz)
        
        assert offset_before != offset_after
    
    def test_extreme_dates(self):
        """Test dates extrêmes"""
        # Date très ancienne
        old_date = datetime(1, 1, 1)
        
        # Date future lointaine
        future_date = datetime(9999, 12, 31)
        
        # Ne doit pas planter
        age_old = age_from_birthdate(old_date)
        days_future = days_until(future_date)
        
        assert isinstance(age_old, int)
        assert isinstance(days_future, int)
        assert age_old > 2000
        assert days_future > 1000
    
    def test_invalid_timezone_handling(self):
        """Test gestion timezone invalide"""
        dt = datetime(2025, 7, 14, 10, 30, 0, tzinfo=pytz.UTC)
        
        # Timezone inexistante
        result = convert_timezone(dt, 'Invalid/Timezone')
        
        # Doit retourner None ou lever exception gérée
        assert result is None or isinstance(result, datetime)
    
    def test_microsecond_precision(self):
        """Test précision microseconde"""
        dt_with_microsec = datetime(2025, 7, 14, 10, 30, 0, 123456)
        
        timestamp = datetime_to_timestamp(dt_with_microsec)
        converted_back = timestamp_to_datetime(timestamp)
        
        # Vérifier préservation précision (dans limites raisonnables)
        diff_microsec = abs(dt_with_microsec.microsecond - converted_back.microsecond)
        assert diff_microsec < 1000  # Moins de 1ms de différence


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
