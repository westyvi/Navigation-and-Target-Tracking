%% gps_data_parser.m
% This script contains parameters and functions for parsing the files
% of project 2 in AEM 667 Navigation & Target Tracking.

%% Setup constants

% Setup WGS84 parameters.

WGS.EQUATORIAL_RADIUS = 6378137; % m
WGS.FLATTENING = 1/298.257223563;
WGS.EARTH_ROTATION = 7292115 * 10^(-11); % rad/s
WGS.EARTH_GRAVITIONAL_CONSTANT = 3.9860050 * 10^(14); % m^3/s^2

WGS.ECCENTRICITY = sqrt(WGS.FLATTENING * (2-WGS.FLATTENING));
WGS.ECCENTRICITY_SQUARED = WGS.FLATTENING * (2-WGS.FLATTENING);
WGS.POLAR_RADIUS = WGS.EQUATORIAL_RADIUS * (1-WGS.FLATTENING);

% Conversion factors

Hz2MHz = 1E-6;
MHz2Hz = 1E6;
s2ns = 1E9;
ns2s = 1E-9;
s2micros = 1E6;
micros2s = 1E-6;
s2ms = 1E3;
ms2s = 1E-3;
dtr = pi / 180;
rtd = 180 / pi;
m2cm = 100;
cm2m = 1 / 100;
m2mm = 1000;
mm2m = 1 / 1000;
ft2m = 0.3048;
m2ft = 1 / 0.3048;
ns2m = SPEED_OF_LIGHT * ns2s;

% GPS SPECIFIC CONSTANTS

L1 = 1575.42e6;         % Hz
L2 = 1227.60e6;
L5 = 1176.45e6;

LAMBDA_L1 = SPEED_OF_LIGHT / L1;     % m
LAMBDA_L2 = SPEED_OF_LIGHT / L2;
LAMBDA_L5 = SPEED_OF_LIGHT / L5;

CA_CODE_RATE = 1.023e6; % chips/s
P_CODE_RATE = 10.23e6;

CA_CHIP_PERIOD = 1 / CA_CODE_RATE;   % s
P_CHIP_PERIOD = 1 / P_CODE_RATE;

CA_CHIP_LENGTH = SPEED_OF_LIGHT / CA_CODE_RATE;  % m
P_CHIP_LENGTH = SPEED_OF_LIGHT / P_CODE_RATE;

CA_CODE_LENGTH = 1023;  % chips


%% ecef2lla

function lla = ecef2lla(xyz, WGS)
% Converts the WGS Earth-Centered, Earth-Fixed (ECEF) Cartesian 
% coordinates to geodetic latitude, longitude, altitude (LLA).

	RAD2DEG = 180/pi;

    % Calculate longitude.
	if ((xyz(1) == 0.0) && (xyz(2) == 0.0))

		long = 0.0;

    else

		long = atan2(xyz(2), xyz(1))*RAD2DEG;

    end

    % Compute altitude and latitude.
	if ((xyz(1) == 0.0) && (xyz(2) == 0.0) && ...
            (xyz(3) == 0.0))
        error('XYZ at center of earth');

    else 
        
        % Compute altitude.
		p = norm([xyz(1) xyz(2)]);
        E = sqrt(WGS.EQUATORIAL_RADIUS^2 - WGS.POLAR_RADIUS^2);
        F = 54 * (WGS.POLAR_RADIUS*xyz(3))^2;
        G = p^2 + (1-WGS.ECCENTRICITY_SQUARED)*xyz(3)^2 ...
            - (WGS.ECCENTRICITY*E)^2;
        c = WGS.ECCENTRICITY^4*F*p^2/G^3;
        s = (1 + WGS.ECCENTRICITY_SQUARED^2*F*p^2/G^3 ...
            + sqrt(c^2 + 2*c))^(1/3);
        P = (F/(3*G^2)) / ((s + (1/s) + 1)^2);
        Q = sqrt(1 + 2*WGS.ECCENTRICITY_SQUARED^2*P);

        k_1 = -P*WGS.ECCENTRICITY_SQUARED*p / (1 + Q);
        k_2 = 0.5*WGS.EQUATORIAL_RADIUS^2*(1 + 1/Q);
        k_3 = -(1 - WGS.ECCENTRICITY_SQUARED)*P*xyz(3)^2 / (Q*(1 + Q));
        k_4 = -0.5*P*p^2;
        r_0 = k_1 + sqrt(k_2 + k_3 + k_4);
        k_5 = (p - WGS.ECCENTRICITY_SQUARED*r_0);

        U = sqrt(k_5^2 + xyz(1)^2);
        V = sqrt(k_5^2 + (1 - WGS.ECCENTRICITY_SQUARED)*xyz(3)^2);
        
        alt = U*(1 - (WGS.POLAR_RADIUS^2 ...
            /(WGS.EQUATORIAL_RADIUS*V)));
        
        % Compute latitude.
        
        z_0 = (WGS.POLAR_RADIUS^2*xyz(3)) ...
            / (WGS.EQUATORIAL_RADIUS*V);
        e_p = (WGS.EQUATORIAL_RADIUS/WGS.POLAR_RADIUS) ...
            * WGS.ECCENTRICITY;
        
        lat = atan((xyz(3) + z_0*(e_p)^2)/p) * RAD2DEG;

	end

    lla = [lat; long; alt];

end


%% lla2ecef

function xyz = lla2ecef(lla, WGS)
% Converts geodetic latitude, longitude, altitude (LLA) to the WGS84
% Earth-Centered, Earth-Fixed (ECEF) Cartesian coordinates.

    east_west_curvature = WGS.EQUATORIAL_RADIUS / sqrt(1 - ...
        WGS.ECCENTRICITY_SQUARED*sin(lla(1)));

    xyz(1, :) = (east_west_curvature + lla(3))...
        *cosd(lla(1)).*cosd(lla(2));
    xyz(2, :) = (east_west_curvature + lla(3))...
        *cosd(lla(1)).*sind(lla(2));
    xyz(3, :) = ((1 - WGS.ECCENTRICITY_SQUARED)...
        *east_west_curvature + lla(3)).*sind(lla(1));

end


%% ecef2ned

function ned = ecef2ned(ecef, ref_ecef, WGS)
% Converts the WGS84 Earth-Centered, Earth-Fixed (ECEF) Cartesian 
% coordinates to the North-East-Down (NED) Cartesian coordinates using
% the reference position and latitude and longitude for the NED frame.
    
   ref_lla = xyz2lla(ref_ecef, WGS);

   delta_ecef = ecef - ref_ecef;

   north = -sind(ref_lla(1)).*cosd(ref_lla(2)).*delta_ecef(1) ...
       - sind(ref_lla(1)).*sind(ref_lla(2)).*delta_ecef(2) ...
       + cosd(ref_lla(1)).*delta_ecef(3);
   
   east = -sind(ref_lla(2)).*delta_ecef(1) ...
       + cosd(ref_lla(2)).*delta_ecef(2);

   down = -cosd(ref_lla(1)).*cosd(ref_lla(2)).*delta_ecef(1) ...
       - cosd(ref_lla(1)).*sind(ref_lla(2)).*delta_ecef(2) ...
       - sind(ref_lla(1)).*delta_ecef(3);

   ned = [north, east, down];

end


%% read_rinex_obs

function [rinex, rec_xyz] = read_rinex_obs(fname, nlines)
% This function reads a RINEX format GPS data file and returns the 
% data in an array.
%
% Colorado Center for Astrodynamics Research
% Copyright 2006 University of Colorado, Boulder

    if (nargin < 2)
        nlines = 1e6;
    end

    % Initialize variables
    rinex_data = [];
    line_count = 0;
    dispLines = 0;

    % Read header
    [ fid, rec_xyz, observables, line_count ] = read_rinex_header(fname);
    num_obs = length(observables);


    % Status
    disp([ 'Parsing RINEX file ' fname ]);

    % Get the first line of the observations.
    current_line = fgetl(fid);
    line_count = line_count + 1;

    % If not at the end of the file, search for the desired information.
    while current_line ~= -1 & line_count < nlines

        % Get the time for this data epoch.
        current_time = [ str2num(current_line(2:3)) ; str2num(current_line(5:6)) ; ...
                str2num(current_line(8:9)) ; str2num(current_line(11:12)) ; ...
                str2num(current_line(14:15)) ; str2num(current_line(17:27)) ]';

        % Error Code
        errorCode = str2num(current_line(28:29));

        % How many SV's are there?
        current_num_sv = str2num(current_line(30:32));

        % Figure out which PRN's there are.
        for ii=1:current_num_sv

            if ii <= 12
                current_prn(ii) = str2num(current_line(31 + 3*ii : 32 + 3*ii));        
                current_prn_type(ii) = current_line(30 + 3*ii);
            else % If there are more than 12, then the next line is used
                if ii == 13
                    current_line = fgetl(fid);
                    line_count = line_count + 1;
                    if rem(line_count, 1) == 0 && dispLines == 1
                        disp([ 'Read ', num2str(line_count) ' lines (Extra Header)' ]);
                    end
                end
                current_prn(ii) = str2num(current_line(31 + 3*(ii-12) : 32 + 3*(ii-12)));
                current_prn_type(ii) = current_line(30 + 3*(ii-12));
            end
        end

        % Get the data for all SV's in this epoch.
        for ii=1:current_num_sv

            if strcmp(current_prn_type(ii),'G')

                % Get the next line.
                current_line = fgetl(fid);
                line_count = line_count + 1;

                if rem(line_count, 1) == 0 && dispLines == 1
                    disp([ 'Read ', num2str(line_count) ' lines (Data - 1st Line)' ]);
                end

                % Check the length of the line and pad it with zeros to
                % make sure it is 80 characters long.
                current_line = check_rinex_line_length(current_line);

                % Get the observables on this line.
                current_obs = [ str2num(current_line(1:14)) ; str2num(current_line(17:30)) ; ...
                        str2num(current_line(33:46)) ; str2num(current_line(49:62)) ; str2num(current_line(65:78)) ];

                % If there are > 5 observables, read another line to get the rest of the observables for this SV.
                if num_obs > 5

                     % Get the next line.
                     current_line = fgetl(fid);
                     line_count = line_count + 1;

                     if rem(line_count, 1) == 0 && dispLines == 1
                        disp([ 'Read ', num2str(line_count) ' lines (Data - 2nd Line)' ]);
                     end

                     % Check the length of the line and pad it with zeros to
                     % make sure it is 80 characters long.
                     current_line = check_rinex_line_length(current_line);

                    % Append the data in this line to the data from previous line.
                    current_obs = [ current_obs ; str2num(current_line(1:14)) ; ...
                                    str2num(current_line(17:30)) ; str2num(current_line(33:46)) ; ...
                                    str2num(current_line(49:62)) ; str2num(current_line(65:78)) ];

                end  % if num_obs > 5

                % If there are > 10 observables, read another line to get the rest of the observables for this SV.
                if num_obs > 10

                     % Get the next line.
                     current_line = fgetl(fid);
                     line_count = line_count + 1;

                     if rem(line_count, 1) == 0 && dispLines == 1
                        disp([ 'Read ', num2str(line_count) ' lines (Data - 3rd Line)' ]);
                     end

                     % Check the length of the line and pad it with zeros to
                     % make sure it is 80 characters long.
                     current_line = check_rinex_line_length(current_line);

                    % Append the data in this line to the data from previous line.
                    current_obs = [ current_obs ; str2num(current_line(1:14)) ; ...
                                    str2num(current_line(17:30)) ; str2num(current_line(33:46)) ; ...
                                    str2num(current_line(49:62)) ; str2num(current_line(65:78)) ];

                end  % if num_obs > 10

                % Format the data for this PRN as Date/Time, PRN, Observations.
                current_data = [ current_time , current_prn(ii) , current_obs' ];


                % Keep only data for the specified PRNs
                if nargin == 3 & PRN_list & isempty(find(PRN_list == current_prn(ii)))
                    continue
                end


                %Append to the master rinex data file.
                rinex_data = [ rinex_data ; current_data ];

            else
                % Get the next line.
                current_line = fgetl(fid);
                line_count = line_count + 1;

                if rem(line_count, 1) == 0 && dispLines == 1
                    disp([ 'Read ', num2str(line_count) ' lines (Glonass - 1st Line)' ]);
                end

                % If there are > 5 observables, read another line to get the rest of the observables for this SV.
                if num_obs > 5

                     % Get the next line.
                     current_line = fgetl(fid);
                     line_count = line_count + 1;

                     if rem(line_count, 1) == 0 && dispLines == 1
                        disp([ 'Read ', num2str(line_count) ' lines (Glonass - 2nd Line)' ]);
                     end
                end

                % If there are > 10 observables, read another line to get the rest of the observables for this SV.
                if num_obs > 10

                     % Get the next line.
                     current_line = fgetl(fid);
                     line_count = line_count + 1;

                     if rem(line_count, 1) == 0 && dispLines == 1
                        disp([ 'Read ', num2str(line_count) ' lines (Glonass - 3rd Line)' ]);
                     end
                end
            end

        end  % for ii=1:current_num_sv

        % Get the next line.
        current_line = fgetl(fid);
        line_count = line_count + 1;

        if rem(line_count, 1) == 0 && dispLines == 1
            disp([ 'Read ', num2str(line_count) ' lines (Header)' ]);
        end

        if current_line ~= -1
            % Check for misplaced headers or Error lines
            while strcmp(current_line(2:3),'  ') || strcmp(current_line(2:3),'ol') || strcmp(current_line(2:3),'eq') || strcmp(current_line(2:3),'ri') || strcmp(current_line(29),'4')
                % Get the next line.
                current_line = fgetl(fid);
                line_count = line_count + 1;
                if rem(line_count, 1) == 0 && dispLines == 1
                    disp([ 'Read ', num2str(line_count) ' lines (Misplaced Header)' ]);
                end
            end
        end

    end  % while current_line ~= -1


    % Convert time format
    [ gpswk, gpssec ] = cal2gpstime(rinex_data(:,1:6));
    rinex.data = [ gpswk gpssec rinex_data(:, 7:end) ];

    % Define columns
    rinex = define_cols(rinex, observables);

    % Convert CP to meters
    rinex = convert_rinex_CP(rinex);


    % Status
    % disp([ 'Total lines: ', num2str(line_count) ]);
    % disp('Finished.');
    % disp(' ');

    fclose(fid);

end

function [ fid, rec_xyz, observables, line_count ] = read_rinex_header( file_name )

    % Initialize the observables variable.
    observables={};
    line_count = 0;

    % Assign a file ID and open the given header file.
    fid=fopen(file_name);

    % If the file does not exist, scream bloody murder!
    if fid == -1
        display('Error!  Header file does not exist.');
    else

        % Set up a flag for when the header file is done.
        end_of_header=0;

        % Get the first line of the file.
        current_line = fgetl(fid);
        line_count = line_count + 1;

        % If not at the end of the file, search for the desired information.
        while end_of_header ~= 1

            % Search for the approximate receiver location line.
            if strfind(current_line,'APPROX POSITION XYZ')

                % Read xyz coordinates into a matrix.
                [rec_xyz] = sscanf(current_line,'%f');
            end

            % Search for the number/types of observables line.
            if strfind(current_line,'# / TYPES OF OBSERV')

                % Read the non-white space characters into a temp variable.
                observables_temp = strsplit(current_line);            
                numObs = str2num(observables_temp{2});

                % Read the number of observables space and then create
                % a matrix containing the actual observables.
                for ii = 1:numObs
                    if ii < 9
                        observables{ii} = observables_temp{ii+2};
                    elseif ii == 9
                        observables_temp2 = strsplit(observables_temp{ii+2},'#'); 
                        observables{ii} = observables_temp2{1};
                    elseif ii == 10
                        % Get the next line of the types of observables
                        current_line = fgetl(fid);
                        line_count = line_count + 1;

                        % Scan and save the line
                        [observables_temp] = strsplit(current_line);
                        observables{ii} = observables_temp{ii-8};
                    else
                        observables{ii} = observables_temp{ii-8};
                    end
                end

            end

            % Get the next line of the header file.
            current_line = fgetl(fid);
            line_count = line_count + 1;

            %Check if this line is at the end of the header file.
            if strfind(current_line,'END OF HEADER')
                end_of_header=1;
            end

        end
    end
end

%% read_rinex_nav

function ephemeris = read_rinex_nav( filename )
% Read the RINEX navigation file.  The header is skipped because
% information in it (a0, a1, iono alpha and beta parameters) is not 
% currently needed for orbit computation.  This can be easily amended to
% include navigation header information by adding lines in the 'while' loop
% where the header is currently skipped.
% 
% Input:        - filename - enter the filename to be read.  If filename
%                            exists, the orbit will be calculated.
% 
% Output:       - ephemeris - Output is a matrix with rows for each PRN and
%                             columns as follows:
% 
%                  col  1:    PRN    ....... satellite PRN          
%                  col  2:    M0     ....... mean anomaly at reference time
%                  col  3:    delta_n  ..... mean motion difference
%                  col  4:    e      ....... eccentricity
%                  col  5:    sqrt(A)  ..... where A is semimajor axis
%                  col  6:    OMEGA  ....... LoAN at weekly epoch
%                  col  7:    i0     ....... inclination at reference time
%                  col  8:    omega  ....... argument of perigee
%                  col  9:    OMEGA_dot  ... rate of right ascension 
%                  col 10:    i_dot  ....... rate of inclination angle
%                  col 11:    Cuc    ....... cosine term, arg. of latitude
%                  col 12:    Cus    ....... sine term, arg. of latitude
%                  col 13:    Crc    ....... cosine term, radius
%                  col 14:    Crs    ....... sine term, radius
%                  col 15:    Cic    ....... cosine term, inclination
%                  col 16:    Cis    ....... sine term, inclination
%                  col 17:    toe    ....... time of ephemeris
%                  col 18:    IODE   ....... Issue of Data Ephemeris
%                  col 19:    GPS_wk ....... GPS week

    fid = fopen(filename);

    if fid == -1
        errordlg(['The file ''' filename ''' does not exist.']);
        return;
    end


    % skip through header
    end_of_header = 0;
    while end_of_header == 0
        current_line = fgetl(fid);
        if strfind(current_line,'END OF HEADER')
            end_of_header=1;
        end
    end

    j = 0;
    while feof(fid) ~= 1
        j = j+1;

        current_line = fgetl(fid);
        % parse epoch line (ignores SV clock bias, drift, and drift rate)
        [PRN, Y, M, D, H, min, sec,af0,af1,af2] = parsef(current_line, {'I2' 'I3' 'I3' 'I3' 'I3' 'I3' ...
                                                      'F5.1','D19.12','D19.12','D19.12'});

        % Broadcast orbit line 1
        current_line = fgetl(fid);
        [IODE Crs delta_n M0] = parsef(current_line, {'D22.12' 'D19.12' 'D19.12' 'D19.12'});

        % Broadcast orbit line 2
        current_line = fgetl(fid);
        [Cuc e Cus sqrtA] = parsef(current_line, {'D22.12' 'D19.12' 'D19.12' 'D19.12'});

        % Broadcast orbit line 3
        current_line = fgetl(fid);
        [toe Cic OMEGA Cis] = parsef(current_line, {'D22.12' 'D19.12' 'D19.12' 'D19.12' 'D19.12'});

        % Broadcast orbit line 4
        current_line = fgetl(fid);
        [i0 Crc omega OMEGA_dot] = parsef(current_line, {'D22.12' 'D19.12' 'D19.12' 'D19.12' 'D19.12'});

        % Broadcast orbit line 5
        current_line = fgetl(fid);
        [i_dot L2_codes GPS_wk L2_dataflag ] = parsef(current_line, {'D22.12' 'D19.12' 'D19.12' 'D19.12' 'D19.12'});

        % Broadcast orbit line 6
        current_line = fgetl(fid);
        [SV_acc SV_health TGD IODC] = parsef(current_line, {'D22.12' 'D19.12' 'D19.12' 'D19.12' 'D19.12'});

        % Broadcast orbit line 7
        current_line = fgetl(fid);
        [msg_trans_t fit_int ] = parsef(current_line, {'D22.12' 'D19.12' 'D19.12'});

        [ gps_week, toc ] = GPSweek(Y,M,D,H,min,sec);

        ephemeris(j,:) = [PRN, M0, delta_n, e, sqrtA, OMEGA, i0, omega, OMEGA_dot, i_dot, Cuc, Cus, Crc, Crs, Cic, Cis, toe, IODE, GPS_wk,toc,af0,af1,af2,TGD];
    end

    fclose(fid);

end

function rinex = define_cols(rinex, observables)

    % Defaults
    rinex.col.WEEK = 1;
    rinex.col.TOW = 2;
    rinex.col.PRN = 3;

    col_offset = 3;

    for ii=1:length(observables)

        switch observables{ii}
            case {'L1'}
                rinex.col.L1 = ii + col_offset;
            case {'L2'}
                rinex.col.L2 = ii + col_offset;
            case {'P1'}
                rinex.col.P1 = ii + col_offset;
            case {'P2'}
                rinex.col.P2 = ii + col_offset;
            case {'C1'}
                rinex.col.C1 = ii + col_offset;    
            case {'D1'}
                rinex.col.D1 = ii + col_offset;
            case {'S1'}
                rinex.col.S1 = ii + col_offset;
            case {'S2'}
                rinex.col.S2 = ii + col_offset;
        end  % switch

    end
end

function [ rinex ] = convert_rinex_CP(rinex)

    set_constants;

    if rinex.col.L1 ~= 0
        rinex.data(:, rinex.col.L1) = rinex.data(:, rinex.col.L1) * LAMBDA_L1;
    end
    % if rinex.col.L2 ~= 0
    %     rinex.data(:, rinex.col.L2) = rinex.data(:, rinex.col.L2) * LAMBDA_L2;
    % end

end

%% parsef

function varargout = parsef(input, format)
%parsef   parse string value using FORTRAN formatting codes
%   [val1,val2,...valn] = parsef(input, format)
%   input is string input value
%   format is cell array of format codes

    global input_
    input_ = input;
    varargout = getvals(1, format, 1);
    clear global input_
    return

end

% this function does the work. you probably don't want to go here
function [output, idx] = getvals(idx, format, reps)
    global input_
    count = 1;
    output = {};
    for k = 1:reps
        odx = 1;
        for i = 1:length(format)
            fmt = format{i};
            switch class(fmt)
            case 'double'
                count = fmt;
            case 'char'
                type = fmt(1);
                if type == 'X'
                    idx = idx+count;
                else
                    [len,cnt] = sscanf(fmt,'%*c%d',1);
                    if cnt ~= 1
                        error(['Invalid format specifier: ''',fmt,'''']);
                    end
                    switch type
                    case {'I','i'}
                        for j = 1:count
                            [val,cnt] = sscanf(input_(idx:min(idx+len-1,end)),'%d',1);
                            if cnt == 1
                                output{odx}(j,k) = val;
                            else
                                output{odx}(j,k) = NaN;
                            end
                            idx = idx+len;
                        end
                    case {'F','f'}
                        for j = 1:count
                            [val,cnt] = sscanf(input_(idx:min(idx+len-1,end)),'%f',1);
                            if cnt == 1
                                output{odx}(j,k) = val;
                            else
                                output{odx}(j,k) = NaN;
                            end
                            idx = idx+len;
                        end
                    case {'E','D','G'}
                        for j = 1:count
                            [val,cnt] = sscanf(input_(idx:min(idx+len-1,end)),'%f%*1[DdEe]%f',2);
                            if cnt == 2
                                output{odx}(j,k) = val(1) * 10^val(2); %#ok<AGROW>
                            elseif cnt == 1
                                output{odx}(j,k) = val;
                            else
                                output{odx}(j,k) = NaN;
                            end
                            idx = idx+len;
                        end
                    case 'A'
                        for j = 1:count
                            output{odx}{j,k} = input_(idx:min(idx+len-1,end));
                            idx = idx+len;
                        end
                    otherwise
                        error(['Invalid format specifier: ''',fmt,'''']);
                    end
                    odx = odx+1;
                end
                count = 1;
            case 'cell'
                [val, idx] = getvals(idx, fmt, count);
                if length(val) == 1
                    output(odx) = val;
                else
                    output{odx} = val;
                end
                odx = odx+1;
                count = 1;
            end
        end
    end

    return
end


%% read_sp3

function [sp3] = read_sp3( filename )
% This function computes reads satellite positions from an SP3 file
%
% Inputs:   filename - filename of SP3 file
%
% Outputs:  sp3_obs = [GPS_week GPS_TOW PRN x y z], sorted by PRN #
%           *note - GPS_TOW is in seconds;  x,y,z are in km
%
% Colorado Center for Astrodynamics Research 
% University of Colorado at Boulder
% October 12, 2006

    % Assign a file ID and open the given header file.
    fid=fopen(filename);

    % If the file does not exist, display warning message
    if fid == -1
        display('Error!  File does not exist.');
    else

        % Go through the header (23 lines long)
        for i = 1:23
            current_line = fgetl(fid);
            % Store the number of satellites in the SP3 file    
            if i == 3
                current_line = current_line(2:length(current_line));
                F = sscanf(current_line,'%u');
                no_sat = F(1);
            end
            i = i + 1;
        end

        % Begin going through times and observations
        end_of_file = 0;
        i = 0; j = 1;
        while end_of_file ~= 1        
            current_line = current_line(2:length(current_line));
            F = sscanf(current_line,'%f');
            % Load GPS Gregorian time into variables
            Y = F(1);
            M = F(2);
            D = F(3);
            H = F(4);
            min = F(5);
            sec = F(6);
            Greg_time(j,:) = [Y M D H min sec];

            % Convert GPS Gregorian time to GPS week and GPS TOW
            [GPS_wk, GPS_TOW] = GPSweek(Y,M,D,H,min,sec);

            % Store satellite PRN and appropriate observations
            for n = 1:no_sat

                % Go to the next line
                current_line = fgetl(fid);
                %current_line = current_line(2:length(current_line));
                current_line = current_line(3:length(current_line));
                F = sscanf(current_line,'%f');

                % Save PRN, positions, and clock error
                PRN = F(1); x = F(2); y = F(3); z = F(4); clk_err = F(5);

                % Create observation vector
                sp3_obs_all(i+n,:) = [GPS_wk GPS_TOW PRN x y z clk_err];
                n = n + 1;
            end

            % Go to next line - check to see if it is the end of file
            current_line = fgetl(fid);
            if strfind(current_line,'EOF')
                end_of_file = 1;
            end

            i = i + n - 1;
            j = j + 1;
        end         
    end

    sp3.data = sp3_obs_all;
    sp3.col.WEEK = 1;
    sp3.col.TOW = 2;
    sp3.col.PRN = 3;
    sp3.col.X = 4;
    sp3.col.Y = 5;
    sp3.col.Z = 6;
    sp3.col.B = 7;

    fclose(fid);

end

%% GPSweek

function [GPS_wk, GPS_sec_wk] = GPSweek(Y, M, D, H, min, sec)
%  Finds GPS week and GPS second of the week based on the
%  input calendar date and time.

    if nargin == 4
	    min = 0;
	    sec = 0;
    end
    
    JD = juliandate([Y, M, D, H, min, sec]);
    GPS_wk = fix((JD-2444244.5)/7);
    GPS_sec_wk = round( ( ((JD-2444244.5)/7)-GPS_wk) * 7 * 24 * 3600);
    
    % Ensure that 1 < GPS_wk < 1024
    GPS_wk = mod( fix((JD-2444244.5)/7), 1024);

end

%% GPSeph2ecef

function ecef = GPSeph2ecef(time, elements)
% Converts a GPS ephemeris to the WGS84 Earth-Centered, Earth-Fixed
% (ECEF) Cartesian coordinates at the specified time.
%
%       VARIABLES               Definition
%   
%       elements.M0     ....... mean anomaly at reference time
%       elements.delta_n  ..... mean motion difference
%       elements.e      ....... eccentricity
%       elements.sqrtA  ....... where A is semimajor axis
%       elements.OMEGA  ....... LoAN at weekly epoch
%       elements.i0     ....... inclination at reference time
%       elements.omega  ....... argument of perigee
%       elements.OMEGA_dot  ... rate of right ascension 
%       elements.i_dot  ....... rate of inclination angle
%       elements.Cuc    ....... cosine term, arg. of latitude
%       elements.Cus    ....... sine term, arg. of latitude
%       elements.Crc    ....... cosine term, radius
%       elements.Crs    ....... sine term, radius
%       elements.Cic    ....... cosine term, inclination
%       elements.Cis    ....... sine term, inclination
%       elements.toe    ....... time of ephemeris

    % Compute the semimajor axis.
    A = elements.sqrtA^2;
    
    % Compute the corrected mean motion (rad/s)
    n = sqrt(MU/A^3) + elements.delta_n;
    
    % Compute the time that has passed since the time of ephemeris.
    delta_time = time - elements.toe;

    % Compue the mean anomaly
    M = elements.M0 + n*delta_time;

    % Compute the eccentric anomaly using Kepler's equation
    E = M;
    ratio = 1;
    
    while abs(ratio) > 10^-8
        E_error = E - elements.e*sin(E) - M;
        E_deriv = 1 - elements.e*cos(E);
        ratio = E_error/E_deriv;
        E = E - ratio;
    end
    E = E + ratio;

    % Compute the true anomaly.
    nu = atan2(sqrt(1-elements.e^2)*sin(E) / (1-elements.e*cos(E)),...
        (cos(E)-elements.e) / (1-elements.e*cos(E)) );

    % Compute the nominal argument of latitude.
    phi = nu + elements.omega;

    % Compute the corrected argument of latitude.
    arg_lat = phi + elements.Cus*sin(2*phi) + elements.Cuc*cos(2*phi);

    % Compute the corrected radius.
    radius = A*(1-elements.e*cos(E)) + elements.Crs*sin(2*phi) ...
        + elements.Crc*cos(2*phi);

    % Compute the corrected inclination.
    incline = elements.i0 + elements.i_dot*delta_time ...
        + elements.Cis*sin(2*phi) + elements.Cic*cos(2*phi);

    % Compute the corrected longitude of the ascending node.
    loan = elements.OMEGA + (elements.OMEGA_dot - EARTH_ROT_RATE)...
        *delta_time - EARTH_ROT_RATE*elements.toe;

    % Compute the orbital in-plane x-y position (ECI)
    x_eci = radius*cos(arg_lat);
    y_eci = radius*sin(arg_lat);

    % Compute the ECEF WGS84 coordinates
    ecef(1, 1) = x_eci*cos(loan) - y_eci*cos(incline)*sin(loan);
    ecef(2, 1) = x_eci*sin(loan) + y_eci*cos(incline)*cos(loan);
    ecef(3, 1) = y_eci*sin(incline);

end

